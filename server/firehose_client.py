"""
FlightAware Firehose Integration for Flight‑Intel – FULL SOURCE

Real‑time (and simulated historical) streaming client used for validation
and enrichment.  This version treats a raw JSON blob on first read as a
successful authentication, eliminating “authentication failed” spam that
appears on the free trial feed.
"""

import os
import ssl
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import struct
import zlib

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ░ CONFIG ░
# ─────────────────────────────────────────────────────────────────────────────
FIREHOSE_HOST = "firehose.flightaware.com"
FIREHOSE_PORT = 1501
FIREHOSE_USERNAME = os.getenv("FIREHOSE_USERNAME")
FIREHOSE_PASSWORD = os.getenv("FIREHOSE_PASSWORD")

# Message types from Firehose
MESSAGE_TYPES = {
    1: "live_position",
    2: "flight_plan",
    3: "departure",
    4: "arrival",
    5: "cancellation",
    7: "flifo",           # Flight Info
    8: "ground_position",
    9: "keep_alive",
    10: "timing_advance", # Foresight predictions
}

# ─────────────────────────────────────────────────────────────────────────────
# ░ DATA CLASS ░
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FirehoseMessage:
    """Parsed Firehose message."""
    message_type: str
    timestamp: datetime
    data: Dict[str, Any]
    pitr: Optional[int] = None  # Point‑in‑time recovery value

# ─────────────────────────────────────────────────────────────────────────────
# ░ CLIENT ░
# ─────────────────────────────────────────────────────────────────────────────
class FirehoseClient:
    """FlightAware Firehose streaming client."""

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected: bool = False
        self.message_buffer: defaultdict[str, List[FirehoseMessage]] = defaultdict(list)
        self.pitr_position: Optional[int] = None

    # ─────────────────────────────────────────────────────────────────────
    async def connect(self) -> bool:
        """Establish an SSL‑wrapped connection to Firehose."""
        try:
            context = ssl.create_default_context()
            self.reader, self.writer = await asyncio.open_connection(
                FIREHOSE_HOST,
                FIREHOSE_PORT,
                ssl=context,
            )

            # Send authentication line
            auth_cmd = f"live username {self.username} password {self.password}\n"
            self.writer.write(auth_cmd.encode())
            await self.writer.drain()

            # First server line
            response = await self.reader.readline()

            # Treat either explicit success or an immediate JSON blob as OK
            if (
                b"authentication successful" in response.lower()
                or response.strip().startswith(b"{")
            ):
                self.connected = True
                logger.info("Firehose authentication successful")

                # Extract PITR position if provided
                if b"pitr" in response:
                    try:
                        token = response.decode().split("pitr")[-1].strip()
                        self.pitr_position = int(token) if token.isdigit() else None
                    except (ValueError, IndexError):
                        self.pitr_position = None

                # Optional layers subscription
                init_args = os.getenv("INIT_CMD_ARGS", "").strip()
                if init_args:
                    sub_cmd = f"{init_args}\n"
                    self.writer.write(sub_cmd.encode())
                    await self.writer.drain()
                    logger.info("Subscribed to Firehose layers: %s", init_args)

                return True

            logger.error("Firehose authentication failed: %s", response.decode().strip())
            await self.disconnect()
            return False

        except Exception as exc:
            logger.error("Firehose connection error: %s", exc)
            self.connected = False
            return False

    # ─────────────────────────────────────────────────────────────────────
    async def disconnect(self) -> None:
        """Close the Firehose connection."""
        if self.writer and not self.writer.is_closing():
            self.writer.close()
            await self.writer.wait_closed()
        self.connected = False

    # ─────────────────────────────────────────────────────────────────────
    async def read_message(self) -> Optional[FirehoseMessage]:
        """Read one length‑prefixed Firehose message and parse to object."""
        if not self.connected or not self.reader:
            return None
        try:
            length_bytes = await self.reader.readexactly(4)
            msg_len = struct.unpack(">I", length_bytes)[0]
            raw = await self.reader.readexactly(msg_len)

            # Decompress if zlib header present
            if raw.startswith(b"\x78\x9c"):
                raw = zlib.decompress(raw)

            payload = json.loads(raw.decode("utf-8"))
            msg_type = MESSAGE_TYPES.get(payload.get("type", 0), "unknown")
            ts = datetime.fromtimestamp(payload.get("timestamp", 0))
            pitr_val = payload.get("pitr")

            return FirehoseMessage(
                message_type=msg_type,
                timestamp=ts,
                data=payload,
                pitr=pitr_val,
            )

        except (asyncio.IncompleteReadError, ConnectionResetError) as exc:
            logger.warning("Firehose stream closed: %s", exc)
            self.connected = False
            return None
        except Exception as exc:
            logger.error("Error reading Firehose message: %s", exc)
            self.connected = False
            return None

    # ─────────────────────────────────────────────────────────────────────
    async def stream_messages(
        self,
        callback: Callable[[FirehoseMessage], Any],
        duration_seconds: Optional[int] = None,
    ) -> None:
        """
        Consume the Firehose stream, invoking `callback` for every message.
        Terminates after `duration_seconds` (if provided) or until connection drops.
        """
        if not self.connected:
            if not await self.connect():
                return

        start = datetime.now()
        try:
            while self.connected:
                if duration_seconds and (datetime.now() - start).total_seconds() > duration_seconds:
                    break

                message = await self.read_message()
                if not message:
                    break  # disconnected

                # Buffer by ident for later lookup
                if "ident" in message.data:
                    self.message_buffer[message.data["ident"]].append(message)

                # Fire user callback
                await callback(message)

        except Exception as exc:
            logger.error("Streaming loop error: %s", exc)
        finally:
            await self.disconnect()

    # ─────────────────────────────────────────────────────────────────────
    async def seek_to_time(self, target: datetime) -> bool:
        """
        Attempt PITR seek for historical trial data.
        """
        if not self.connected or not self.writer:
            return False
        try:
            cmd = f"pitr {int(target.timestamp())}\n"
            self.writer.write(cmd.encode())
            await self.writer.drain()
            resp = await self.reader.readline()
            if b"successful" in resp.lower():
                logger.info("PITR seek to %s successful", target)
                return True
            logger.error("PITR seek failed: %s", resp.decode().strip())
            return False
        except Exception as exc:
            logger.error("PITR seek error: %s", exc)
            return False

    # ─────────────────────────────────────────────────────────────────────
    def get_flight_data(self, ident: str) -> List[FirehoseMessage]:
        """Return all buffered messages for a given ident."""
        return self.message_buffer.get(ident.upper(), [])

# ─────────────────────────────────────────────────────────────────────────────
# ░ VALIDATOR HELPER ░
# ─────────────────────────────────────────────────────────────────────────────
class FirehoseValidator:
    """High‑level helper that uses FirehoseClient for flight validation."""

    def __init__(self) -> None:
        if FIREHOSE_USERNAME and FIREHOSE_PASSWORD:
            self.client = FirehoseClient(FIREHOSE_USERNAME, FIREHOSE_PASSWORD)
        else:
            logger.warning("Firehose credentials not set – Firehose validator disabled.")
            self.client = None

    # ---------------------------------------------------------------------
    async def validate_flight(self, flight_no: str, date_str: str) -> Optional[Dict]:
        """
        Validate flight using Firehose.  Routes to realtime stream unless the
        date is >2 days in the past or future (historical unavailable in trial).
        """
        if not self.client:
            return None

        try:
            flight_date = datetime.strptime(date_str, "%m/%d/%Y")
        except ValueError:
            logger.debug("Firehose validator: invalid date %s", date_str)
            return None

        if abs((datetime.now() - flight_date).days) > 2:
            logger.info("Historical Firehose validation not available on trial.")
            return None

        return await self._validate_realtime(flight_no)

    # ---------------------------------------------------------------------
    async def _validate_realtime(self, flight_no: str) -> Optional[Dict]:
        """Watch the live stream briefly and extract FLIFO/depart/arrive info."""
        if not self.client:
            return None

        found_msgs: List[FirehoseMessage] = []
        target_ident = flight_no.upper()

        async def collector(msg: FirehoseMessage) -> None:
            if msg.data.get("ident", "").upper() == target_ident:
                found_msgs.append(msg)
                if len(found_msgs) > 5 and self.client.connected:
                    self.client.connected = False  # stop streaming

        await self.client.stream_messages(collector, duration_seconds=30)

        if not found_msgs:
            return None

        return self._summarise(found_msgs)

    # ---------------------------------------------------------------------
    @staticmethod
    def _summarise(messages: List[FirehoseMessage]) -> Dict:
        """Convert a list of FirehoseMessages into an enrichment dict."""
        summary: Dict[str, Any] = {
            "source": "firehose",
            "confidence": 0.99,
            "data": {},
        }

        messages.sort(key=lambda m: m.timestamp)
        for msg in messages:
            d = msg.data
            if msg.message_type == "flifo":
                summary["data"].update(
                    {
                        "origin": d.get("origin"),
                        "destination": d.get("destination"),
                        "aircraft_type": d.get("aircrafttype"),
                        "route": d.get("route"),
                    }
                )
            elif msg.message_type == "departure":
                summary["data"]["actual_departure_time"] = msg.timestamp.isoformat()
            elif msg.message_type == "arrival":
                summary["data"]["actual_arrival_time"] = msg.timestamp.isoformat()
        return summary

# ─────────────────────────────────────────────────────────────────────────────
# ░ PUBLIC FUNCTION ░
# ─────────────────────────────────────────────────────────────────────────────
async def validate_with_firehose(flight: Dict) -> Optional[Dict]:
    """
    Convenience wrapper used by flight_intel_patch.
    """
    if not flight.get("flight_no") or not flight.get("date"):
        return None

    validator = FirehoseValidator()
    if not validator.client:
        return None

    try:
        return await validator.validate_flight(flight["flight_no"], flight["date"])
    except Exception as exc:
        logger.error("Firehose validation helper error: %s", exc)
        return None
