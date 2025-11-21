from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from dotenv import load_dotenv

from logging_utils import log_event

load_dotenv()

logger = logging.getLogger("flightintel.fr24")

FR24_BASE_URL = "https://api.flightradar24.com/common/v1"
FR24_RATE = float(os.getenv("FR24_MAX_RPS", "0.5"))  # ~30/min
FR24_BURST = int(os.getenv("FR24_BURST", "5"))


class _AsyncTokenBucket:
    def __init__(self, rate: float, burst: int):
        self.rate = float(rate)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.t = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self, n: float = 1.0):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if self.tokens < n:
                wait = (n - self.tokens) / self.rate
                await asyncio.sleep(wait)
                self.tokens = 0.0
            else:
                self.tokens -= n


class FR24Client:
    """
    FlightRadar24 API client with:
      - Token-bucket limiter
      - Small response cache
      - 429 retry with backoff
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=30,
        )
        self._limiter = _AsyncTokenBucket(rate=FR24_RATE, burst=FR24_BURST)
        self._cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}
        self._cache_ttl = 30.0
        self._cache_max_size = 200

    async def __aenter__(self) -> "FR24Client":
        timeout = aiohttp.ClientTimeout(total=15, connect=4)
        self._session = aiohttp.ClientSession(
            headers={
                "Accept": "application/json",
                "User-Agent": "Flight-Intel/2.3",
            },
            connector=self._connector,
            timeout=timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    def _get_from_cache(self, ident: str, date: str) -> Optional[Any]:
        key = (ident.upper(), date)
        if key in self._cache:
            ts, data = self._cache[key]
            if time.perf_counter() - ts < self._cache_ttl:
                log_event(
                    logger,
                    "fr24_cache_hit",
                    ident=ident,
                    date=date,
                )
                return data
            del self._cache[key]
        return None

    def _put_in_cache(self, ident: str, date: str, data: Any) -> None:
        key = (ident.upper(), date)
        self._cache[key] = (time.perf_counter(), data)
        if len(self._cache) > self._cache_max_size:
            oldest_key = min(self._cache.items(), key=lambda x: x[1][0])[0]
            del self._cache[oldest_key]

    async def _do_get(
        self, url: str, params: Dict[str, Any]
    ) -> Tuple[int, Any, float]:
        await self._limiter.acquire()
        t0 = time.perf_counter()
        max_attempts = 3
        status = 0
        body: Any = None

        for attempt in range(max_attempts):
            try:
                async with self._session.get(url, params=params) as r:
                    elapsed = time.perf_counter() - t0
                    status = r.status
                    try:
                        body = await r.json()
                    except Exception:
                        body = await r.text()

                    log_event(
                        logger,
                        "fr24_http_call",
                        provider="fr24",
                        endpoint=url,
                        status_code=status,
                        duration_ms=int(elapsed * 1000),
                        attempt=attempt + 1,
                    )

                    if status == 429 and attempt < max_attempts - 1:
                        retry_after = r.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = float(retry_after) + 1.0
                            except ValueError:
                                wait_time = 3.0
                        else:
                            wait_time = (2**attempt) + random.uniform(0.5, 2.0)

                        log_event(
                            logger,
                            "fr24_rate_limited",
                            level=logging.WARNING,
                            provider="fr24",
                            retry_after=retry_after,
                            wait_seconds=wait_time,
                            attempt=attempt + 1,
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    if status != 200:
                        error_text = (
                            body if isinstance(body, str) else json.dumps(body)[:200]
                        )
                        log_event(
                            logger,
                            "fr24_error",
                            level=logging.ERROR,
                            status_code=status,
                            error_preview=error_text,
                        )

                    return status, body, elapsed

            except asyncio.TimeoutError:
                log_event(
                    logger,
                    "fr24_timeout",
                    level=logging.ERROR,
                    attempt=attempt + 1,
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.0 + random.uniform(0, 0.5))
                    continue
                return 504, None, time.perf_counter() - t0

            except Exception as e:
                log_event(
                    logger,
                    "fr24_exception",
                    level=logging.ERROR,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.0)
                    continue
                return 500, None, time.perf_counter() - t0

        return status, body, time.perf_counter() - t0

    async def search(
        self,
        flight_no: str,
        date_mmddyyyy: str,
        *,
        origin_hint: Optional[str] = None,
        dest_hint: Optional[str] = None,
    ) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Search for flight data on FR24.

        Returns:
        - Single dict if only one match OR data is complete
        - List of dicts if multiple matches AND data is incomplete
        - None if no matches found
        """
        if not self._session or not self._api_key:
            log_event(
                logger,
                "fr24_not_initialized",
                level=logging.WARNING,
            )
            return None

        ident = flight_no.strip().upper()

        cached = self._get_from_cache(ident, date_mmddyyyy)
        if cached is not None:
            return cached

        params = {
            "query": ident,
            "fetchBy": "flight",
            "limit": 25,
            "token": self._api_key,
        }

        url = f"{FR24_BASE_URL}/flight/list.json"
        status, body, _elapsed = await self._do_get(url, params)

        if status != 200 or not body:
            return None

        try:
            flights = (
                body.get("result", {})
                .get("response", {})
                .get("data", [])
            )
            if not flights:
                log_event(
                    logger,
                    "fr24_no_flights",
                    ident=ident,
                )
                return None

            log_event(
                logger,
                "fr24_matches_found",
                ident=ident,
                count=len(flights),
            )

            def _matches_ident(flight_data: Dict[str, Any]) -> bool:
                for key in ("identification", "flight", "airline"):
                    if key in flight_data:
                        obj = flight_data[key]
                        if isinstance(obj, dict):
                            number = obj.get("number", {})
                            if isinstance(number, dict):
                                default_num = number.get("default", "").upper()
                                if default_num == ident:
                                    return True
                            callsign = obj.get("callsign", "").upper()
                            if callsign == ident:
                                return True
                return False

            matching_flights = [f for f in flights if _matches_ident(f)]

            has_incomplete_data = not all([origin_hint, dest_hint])

            if not matching_flights:
                log_event(
                    logger,
                    "fr24_no_exact_match",
                    ident=ident,
                )
                result = flights[0]
                self._put_in_cache(ident, date_mmddyyyy, result)
                return result

            if has_incomplete_data and len(matching_flights) > 1:
                log_event(
                    logger,
                    "fr24_multi_match",
                    ident=ident,
                    count=len(matching_flights),
                )
                self._put_in_cache(ident, date_mmddyyyy, matching_flights)
                return matching_flights

            result = matching_flights[0]
            log_event(
                logger,
                "fr24_single_match",
                ident=ident,
            )
            self._put_in_cache(ident, date_mmddyyyy, result)
            return result

        except Exception as e:
            log_event(
                logger,
                "fr24_parse_error",
                level=logging.ERROR,
                error=str(e),
            )
            return None
