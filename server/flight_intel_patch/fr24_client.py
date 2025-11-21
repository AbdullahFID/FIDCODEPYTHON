from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp

from .config import FR24_BASE_URL
from .utils import _AsyncTokenBucket

logger = logging.getLogger("flight-validator.fr24")


class FR24Client:
    """
    FlightRadar24 API client with similar resilience to AeroAPIClient.

    Endpoint used:
      - GET /flight/list.json?query={IDENT}&fetchBy=flight&limit=25&token={API_KEY}
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
        # Rate limiting: 30/min ≈ 0.5/sec with burst capacity of 5
        self._limiter = _AsyncTokenBucket(rate=0.5, burst=5)

        # Response cache: (ident, date) -> (timestamp, data)
        self._cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}
        self._cache_ttl = 30.0  # seconds
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
        """Check response cache for recent identical request."""
        key = (ident.upper(), date)
        if key in self._cache:
            ts, data = self._cache[key]
            if time.perf_counter() - ts < self._cache_ttl:
                logger.info(f"FR24 cache HIT for {ident} on {date}")
                return data
            else:
                del self._cache[key]
        return None

    def _put_in_cache(self, ident: str, date: str, data: Any) -> None:
        """Store successful response in cache."""
        key = (ident.upper(), date)
        self._cache[key] = (time.perf_counter(), data)

        if len(self._cache) > self._cache_max_size:
            oldest_key = min(self._cache.items(), key=lambda x: x[1][0])[0]
            del self._cache[oldest_key]

    async def _do_get(
        self, url: str, params: Dict[str, Any]
    ) -> Tuple[int, Any, float]:
        """
        Execute GET request with rate limiting, caching, and retry logic.
        Returns: (status_code, response_data, elapsed_time)
        """
        if not self._session:
            return 503, None, 0.0

        await self._limiter.acquire()

        t0 = time.perf_counter()
        max_attempts = 3
        status: int = 0
        body: Any = None

        for retry in range(max_attempts):
            try:
                async with self._session.get(url, params=params) as r:
                    elapsed = time.perf_counter() - t0
                    status = r.status

                    try:
                        body = await r.json()
                    except Exception:
                        body = await r.text()

                    logger.info(
                        f"FR24 GET {url.split('/')[-1]} "
                        f"status={status} took={elapsed:.2f}s attempt={retry+1}/{max_attempts}"
                    )

                    if status == 429 and retry < max_attempts - 1:
                        retry_after = r.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = float(retry_after) + 1.0
                            except ValueError:
                                wait_time = 3.0
                        else:
                            wait_time = (2 ** retry) + random.uniform(0.5, 2.0)

                        logger.warning(
                            f"FR24 429 rate limit hit - waiting {wait_time:.1f}s before retry"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    if status != 200:
                        error_text = body if isinstance(body, str) else json.dumps(body)[:200]
                        logger.error(f"FR24 error {status}: {error_text}")

                    return status, body, elapsed

            except asyncio.TimeoutError:
                logger.error(f"FR24 timeout on attempt {retry+1}/{max_attempts}")
                if retry < max_attempts - 1:
                    await asyncio.sleep(1.0 + random.uniform(0, 0.5))
                    continue
                return 504, None, time.perf_counter() - t0

            except Exception as e:
                logger.error(f"FR24 exception on attempt {retry+1}/{max_attempts}: {e}")
                if retry < max_attempts - 1:
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
            logger.warning("FR24 client not initialized or missing API key")
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

        status, body, elapsed = await self._do_get(url, params)

        if status != 200 or not body:
            return None

        try:
            flights = body.get("result", {}).get("response", {}).get("data", [])
            if not flights:
                logger.info(f"FR24 no flights found for {ident}")
                return None

            logger.info(f"FR24 found {len(flights)} potential matches for {ident}")

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
                logger.warning(f"FR24 no exact match for {ident}, using first result")
                result = flights[0]
                self._put_in_cache(ident, date_mmddyyyy, result)
                return result

            if has_incomplete_data and len(matching_flights) > 1:
                logger.info(
                    f"FR24 returning {len(matching_flights)} options for {ident} "
                    f"(incomplete data: origin={bool(origin_hint)}, dest={bool(dest_hint)})"
                )
                self._put_in_cache(ident, date_mmddyyyy, matching_flights)
                return matching_flights

            result = matching_flights[0]
            logger.info(f"FR24 returning single best match for {ident}")
            self._put_in_cache(ident, date_mmddyyyy, result)
            return result

        except Exception as e:
            logger.error(f"FR24 response parsing error: {e}")
            return None
