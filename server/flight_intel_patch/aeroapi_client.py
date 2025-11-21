from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp

from .config import AEROAPI_BASE_URL, AEROAPI_MAX_RPS, AEROAPI_BURST
from .utils import (
    FLIGHT_PATTERN,
    _AsyncTokenBucket,
    _iso_day_window_utc,
    _normalize_airport_fast,
    _split_ident,
)

logger = logging.getLogger("flight-validator.aeroapi")

# Lightweight response cache for identical AeroAPI GETs
_AERO_RESP_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Tuple[float, Any]] = {}
_AERO_CACHE_TTL = 30.0  # seconds


class AeroAPIClient:
    """
    FlightAware AeroAPI v4 client (read-only) with robust fallbacks.

    Endpoints used:
      - GET /flights/{ident}?start={iso}&end={iso}
      - GET /history/flights/{ident}?start={iso}&end={iso}
      - GET /schedules/{date_start}/{date_end}?airline={DL}&flight_number={9013}
      - GET /schedules/{date_start}/{date_end}?airline={DL}&destination={MCO}[&origin=MSP]
    """

    def __init__(self, api_key: str) -> None:
        self._headers = {"x-apikey": api_key, "Accept": "application/json"}
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=30,
        )
        self._limiter = _AsyncTokenBucket(AEROAPI_MAX_RPS, AEROAPI_BURST)

    async def __aenter__(self) -> "AeroAPIClient":
        timeout = aiohttp.ClientTimeout(total=12, connect=3)
        self._session = aiohttp.ClientSession(
            headers=self._headers, connector=self._connector, timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    async def _do_get(self, path: str, params: Dict[str, Any]) -> Tuple[int, Any, float]:
        """GET with rate limit, small cache, and 429 backoff."""
        if not self._session:
            return 503, None, 0.0

        await self._limiter.acquire()
        url = f"{AEROAPI_BASE_URL}{path}"
        key = (path, tuple(sorted((k, str(v)) for k, v in params.items())))
        now = time.perf_counter()

        # Response de-dupe cache
        hit = _AERO_RESP_CACHE.get(key)
        if hit and (now - hit[0] < _AERO_CACHE_TTL):
            status, body = 200, hit[1]
            logger.info(f"AeroAPI GET (cache) {path} params={params} status={status}")
            return status, body, 0.0

        t0 = now
        attempts = 3
        status: int = 0
        body: Any = None
        for attempt in range(attempts):
            async with self._session.get(url, params=params) as r:
                elapsed = time.perf_counter() - t0
                status = r.status
                try:
                    body = await r.json()
                except Exception:
                    body = await r.text()
                logger.info(
                    f"AeroAPI GET {path} params={params} status={status} took={elapsed:.2f}s"
                )

                if status == 429 and attempt < attempts - 1:
                    ra = r.headers.get("Retry-After")
                    if ra:
                        try:
                            await asyncio.sleep(float(ra) + 0.5)
                        except Exception:
                            await asyncio.sleep(2.0)
                    else:
                        await asyncio.sleep(2.0 + random.random())
                    continue

                if status == 200:
                    _AERO_RESP_CACHE[key] = (time.perf_counter(), body)
                    if path.startswith("/schedules") and isinstance(body, dict):
                        if "scheduled" in body:
                            logger.info(
                                f"   Found {len(body['scheduled'])} scheduled flights"
                            )
                            if body["scheduled"]:
                                first = body["scheduled"][0]
                                logger.info(
                                    f"   First flight: {first.get('ident_iata')} from "
                                    f"{first.get('origin_iata')} to {first.get('destination_iata')}"
                                )
                return status, body, elapsed
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
        Returns flight data:
        - Single dict if only one match OR all data is complete
        - List of dicts if multiple matches AND data is incomplete
        - None if no matches found
        """
        if not self._session:
            return None

        ident = (flight_no or "").strip().upper()
        if not ident or not FLIGHT_PATTERN.match(ident):
            logger.warning(f"AeroAPI: invalid flight ident '{ident}'")
            return None

        # Parse date & decide endpoint
        try:
            m, d, y = [int(x) for x in date_mmddyyyy.split("/")]
            flight_dt = datetime(y, m, d, tzinfo=timezone.utc)
        except Exception:
            logger.error(f"AeroAPI: bad date '{date_mmddyyyy}'")
            return None

        day_start_iso, day_end_iso = _iso_day_window_utc(
            flight_dt.year, flight_dt.month, flight_dt.day
        )
        days_delta = (flight_dt.date() - datetime.utcnow().date()).days

        has_incomplete_data = not all([origin_hint, dest_hint])

        S_PARAMS = {
            "include_codeshares": "false",
            "max_pages": "1",
        }

        last_error_text: Optional[str] = None

        def _extract_all_items(body: Any) -> List[Dict[str, Any]]:
            if not isinstance(body, dict):
                return []

            flights_list: List[Dict[str, Any]] = []

            if "scheduled" in body and isinstance(body["scheduled"], list):
                flights_list = body["scheduled"]
            elif "flights" in body and isinstance(body["flights"], list):
                flights_list = body["flights"]
            elif "data" in body and isinstance(body["data"], list):
                flights_list = body["data"]
            else:
                for key, val in body.items():
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        flights_list = val
                        break

            if not flights_list:
                return []

            def _matches_ident(f: Dict[str, Any]) -> bool:
                for k in ("ident", "ident_iata", "ident_icao", "flight_number"):
                    v = f.get(k)
                    if isinstance(v, str):
                        v_clean = v.strip().upper()
                        if v_clean == ident:
                            return True
                        if v_clean.endswith(ident[2:]) and len(v_clean) > len(ident):
                            return True

                airline, number, _sfx = _split_ident(ident)
                if number:
                    fn = str(f.get("flight_number") or "").strip()
                    if fn == number:
                        return True

                return False

            matching = [item for item in flights_list if _matches_ident(item)]
            return matching

        def _extract_first_item(body: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(body, dict):
                return None
            flights_list: List[Dict[str, Any]] = []

            if "scheduled" in body and isinstance(body["scheduled"], list):
                flights_list = body["scheduled"]
            elif "flights" in body and isinstance(body["flights"], list):
                flights_list = body["flights"]
            elif "data" in body and isinstance(body["data"], list):
                flights_list = body["data"]
            else:
                for key, val in body.items():
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        flights_list = val
                        break

            if not flights_list:
                return None

            def _matches_ident(f: Dict[str, Any]) -> bool:
                for k in ("ident", "ident_iata", "ident_icao", "flight_number"):
                    v = f.get(k)
                    if isinstance(v, str):
                        v_clean = v.strip().upper()
                        if v_clean == ident:
                            return True
                        if v_clean.endswith(ident[2:]) and len(v_clean) > len(ident):
                            return True

                airline, number, _sfx = _split_ident(ident)
                if number:
                    fn = str(f.get("flight_number") or "").strip()
                    if fn == number:
                        return True

                return False

            for item in flights_list:
                if _matches_ident(item):
                    return item
            return flights_list[0]

        phaseA: List[Tuple[str, Dict[str, Any]]] = []

        if days_delta < -10:
            phaseA.append(
                (
                    f"/history/flights/{ident}",
                    {"start": day_start_iso, "end": day_end_iso},
                )
            )
        elif -10 <= days_delta <= 2:
            phaseA.append(
                (f"/flights/{ident}", {"start": day_start_iso, "end": day_end_iso})
            )
        else:
            airline, number, _suffix = _split_ident(ident)
            if number:
                params = {"flight_number": number, **S_PARAMS}
                if airline:
                    params["airline"] = airline
                phaseA.append(
                    (
                        f"/schedules/{flight_dt.strftime('%Y-%m-%d')}/{(flight_dt + timedelta(days=1)).strftime('%Y-%m-%d')}",
                        params,
                    )
                )

        def _have(prefix: str, arr: List[Tuple[str, Dict[str, Any]]]) -> bool:
            return any(p.startswith(prefix) for p, _ in arr)

        if not _have("/flights/", phaseA):
            phaseA.append(
                (f"/flights/{ident}", {"start": day_start_iso, "end": day_end_iso})
            )
        if not _have("/history/", phaseA):
            phaseA.append(
                (
                    f"/history/flights/{ident}",
                    {"start": day_start_iso, "end": day_end_iso},
                )
            )

        for path, params in phaseA:
            status, body, _elapsed = await self._do_get(path, params)

            if status == 400:
                err_txt = body if isinstance(body, str) else json.dumps(body)[:200]
                logger.error(f"AeroAPI 400 on {path} params={params} err={err_txt}")
                last_error_text = err_txt
                continue
            if status in (404, 204):
                continue
            if status != 200:
                last_error_text = (
                    body if isinstance(body, str) else json.dumps(body)[:200]
                )
                continue

            if has_incomplete_data:
                items = _extract_all_items(body)
                if len(items) > 1:
                    logger.info(
                        f"   📋 Found {len(items)} matching flights - returning all "
                        f"due to incomplete data"
                    )
                    return items
                elif len(items) == 1:
                    return items[0]
            else:
                item = _extract_first_item(body)
                if item:
                    return item

        sched_path = f"/schedules/{flight_dt.strftime('%Y-%m-%d')}/{(flight_dt + timedelta(days=1)).strftime('%Y-%m-%d')}"
        airline, number, _suffix = _split_ident(ident)
        route_tries: List[Dict[str, Any]] = []
        if dest_hint:
            p = {"airline": airline or "DL", "destination": dest_hint, **S_PARAMS}
            if origin_hint:
                p["origin"] = origin_hint
            route_tries.append(p)
        if origin_hint and not dest_hint:
            route_tries.append(
                {"airline": airline or "DL", "origin": origin_hint, **S_PARAMS}
            )

        for params in route_tries:
            status, body, _elapsed = await self._do_get(sched_path, params)

            if status == 400:
                err_txt = body if isinstance(body, str) else json.dumps(body)[:200]
                logger.error(
                    f"AeroAPI 400 on {sched_path} params={params} err={err_txt}"
                )
                last_error_text = err_txt
                continue
            if status in (404, 204):
                continue
            if status != 200:
                last_error_text = (
                    body if isinstance(body, str) else json.dumps(body)[:200]
                )
                continue

            if has_incomplete_data:
                items = _extract_all_items(body)
                if len(items) > 1:
                    logger.info(
                        f"   📋 Found {len(items)} matching flights - returning all "
                        f"due to incomplete data"
                    )
                    return items
                elif len(items) == 1:
                    return items[0]
            else:
                item = _extract_first_item(body)
                if item:
                    return item

        if last_error_text:
            logger.error(f"AeroAPI no data; last error: {last_error_text}")
        return None
