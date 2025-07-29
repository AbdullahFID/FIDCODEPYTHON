"""
FlightAware Firehose Integration for Flight‑Intel – ENHANCED VERSION

Real‑time streaming client with enhanced features:
- Better scheduled time extraction from FLIFO messages
- Intelligent message caching with TTL
- Enhanced data extraction for all message types
- Automatic reconnection with exponential backoff
- Support for extracting gate/terminal information
- Better handling of flight plan and route data
"""

import os
import ssl
import json
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import struct
import zlib
import re
from enum import Enum

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ░ CONFIG ░
# ─────────────────────────────────────────────────────────────────────────────
FIREHOSE_HOST = "firehose.flightaware.com"
FIREHOSE_PORT = 1501
FIREHOSE_USERNAME = os.getenv("FIREHOSE_USERNAME")
FIREHOSE_PASSWORD = os.getenv("FIREHOSE_PASSWORD")

# Enhanced message types
MESSAGE_TYPES = {
    1: "live_position",
    2: "flight_plan",
    3: "departure",
    4: "arrival",
    5: "cancellation",
    6: "surface_offblock",  # Push back from gate
    7: "flifo",             # Flight Info (richest data)
    8: "ground_position",
    9: "keep_alive",
    10: "timing_advance",   # Foresight predictions
    11: "extended_flightinfo",
    12: "surface_onblock",  # Arrival at gate
}

# Cache configuration
CACHE_TTL_MINUTES = 30
MAX_CACHE_SIZE = 1000
MAX_MESSAGES_PER_FLIGHT = 50

# ─────────────────────────────────────────────────────────────────────────────
# ░ DATA CLASSES ░
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FirehoseMessage:
    """Enhanced Firehose message with additional metadata."""
    message_type: str
    timestamp: datetime
    data: Dict[str, Any]
    pitr: Optional[int] = None
    raw_type: Optional[int] = None
    ident: Optional[str] = None
    
    def __post_init__(self):
        # Extract ident for easier access
        if not self.ident and "ident" in self.data:
            self.ident = self.data["ident"].upper()


@dataclass 
class FlightCache:
    """Cache entry for a flight with TTL and message limit."""
    ident: str
    messages: List[FirehoseMessage] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    summary: Optional[Dict[str, Any]] = None
    
    def add_message(self, msg: FirehoseMessage) -> None:
        """Add message with size limit."""
        self.messages.append(msg)
        if len(self.messages) > MAX_MESSAGES_PER_FLIGHT:
            # Keep most important message types
            important_types = {"flifo", "departure", "arrival", "flight_plan", "cancellation"}
            self.messages = [
                m for m in self.messages[-MAX_MESSAGES_PER_FLIGHT:]
                if m.message_type in important_types
            ] + [
                m for m in self.messages[-MAX_MESSAGES_PER_FLIGHT:]
                if m.message_type not in important_types
            ][:MAX_MESSAGES_PER_FLIGHT//2]
        self.last_updated = datetime.utcnow()
        
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return (datetime.utcnow() - self.last_updated) > timedelta(minutes=CACHE_TTL_MINUTES)


class ConnectionState(Enum):
    """Connection state management."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    STREAMING = "streaming"
    ERROR = "error"

# ─────────────────────────────────────────────────────────────────────────────
# ░ ENHANCED CLIENT ░
# ─────────────────────────────────────────────────────────────────────────────
class FirehoseClient:
    """Enhanced FlightAware Firehose streaming client."""

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.state = ConnectionState.DISCONNECTED
        self.cache: Dict[str, FlightCache] = {}
        self.pitr_position: Optional[int] = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.base_reconnect_delay = 2  # seconds
        self._seen_idents: Set[str] = set()  # Track unique flights seen

    # ─────────────────────────────────────────────────────────────────────
    async def connect(self, retry: bool = True) -> bool:
        """Establish connection with optional retry logic."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.state = ConnectionState.CONNECTING
                context = ssl.create_default_context()
                
                self.reader, self.writer = await asyncio.wait_for(
                    asyncio.open_connection(FIREHOSE_HOST, FIREHOSE_PORT, ssl=context),
                    timeout=30
                )
                
                self.state = ConnectionState.CONNECTED
                
                # Send authentication
                auth_cmd = f"live username {self.username} password {self.password}\n"
                self.writer.write(auth_cmd.encode())
                await self.writer.drain()

                # Read authentication response
                response = await asyncio.wait_for(self.reader.readline(), timeout=10)

                # Accept either explicit success or immediate JSON
                if (
                    b"authentication successful" in response.lower()
                    or response.strip().startswith(b"{")
                ):
                    self.state = ConnectionState.AUTHENTICATED
                    logger.info("Firehose authentication successful")
                    
                    # Reset reconnect counter on success
                    self.reconnect_attempts = 0

                    # Extract PITR if available
                    if b"pitr" in response:
                        self._extract_pitr(response)

                    # Subscribe to additional layers if configured
                    await self._subscribe_layers()
                    
                    self.state = ConnectionState.STREAMING
                    return True

                logger.error("Authentication failed: %s", response.decode().strip())
                self.state = ConnectionState.ERROR
                await self.disconnect()
                
            except asyncio.TimeoutError:
                logger.error("Connection/authentication timeout")
                self.state = ConnectionState.ERROR
                
            except Exception as exc:
                logger.error("Connection error: %s", exc)
                self.state = ConnectionState.ERROR
                
            if not retry:
                return False
                
            # Exponential backoff
            delay = self.base_reconnect_delay * (2 ** self.reconnect_attempts)
            logger.info("Retrying connection in %d seconds...", delay)
            await asyncio.sleep(delay)
            self.reconnect_attempts += 1
            
        logger.error("Max reconnection attempts reached")
        return False

    # ─────────────────────────────────────────────────────────────────────
    async def disconnect(self) -> None:
        """Gracefully close connection."""
        self.state = ConnectionState.DISCONNECTED
        if self.writer and not self.writer.is_closing():
            self.writer.close()
            await self.writer.wait_closed()
        self.reader = None
        self.writer = None

    # ─────────────────────────────────────────────────────────────────────
    def _extract_pitr(self, response: bytes) -> None:
        """Extract PITR position from response."""
        try:
            decoded = response.decode()
            if "pitr" in decoded:
                match = re.search(r'pitr\s+(\d+)', decoded)
                if match:
                    self.pitr_position = int(match.group(1))
                    logger.info("PITR position: %d", self.pitr_position)
        except Exception as e:
            logger.warning("Failed to extract PITR: %s", e)

    # ─────────────────────────────────────────────────────────────────────
    async def _subscribe_layers(self) -> None:
        """Subscribe to configured Firehose layers."""
        init_args = os.getenv("INIT_CMD_ARGS", "").strip()
        if init_args:
            cmd = f"{init_args}\n"
            self.writer.write(cmd.encode())
            await self.writer.drain()
            logger.info("Subscribed to layers: %s", init_args)

    # inside class FirehoseClient
    async def read_message(self) -> Optional[FirehoseMessage]:
        """Read one Firehose message. Supports newline keep‑alives, line‑JSON, and 4‑byte length framing."""
        if self.state != ConnectionState.STREAMING or not self.reader:
            return None

        try:
            # 60s: expect either a keep‑alive '\n' or the start of a frame
            b0 = await asyncio.wait_for(self.reader.readexactly(1), timeout=60)

            # bare newline = keep‑alive
            if b0 in (b"\n", b"\r"):
                return None

            # Some feeds emit line‑delimited JSON (first byte is '{' or '[')
            if b0 in (b"{", b"["):
                rest = await asyncio.wait_for(self.reader.readline(), timeout=30)
                raw = b0 + rest
            else:
                # Classic 4‑byte big‑endian length framing
                b123 = await asyncio.wait_for(self.reader.readexactly(3), timeout=30)
                msg_len = struct.unpack(">I", b0 + b123)[0]

                # sanity cap
                if msg_len > 1_048_576:  # 1 MB
                    await asyncio.wait_for(self.reader.readexactly(msg_len), timeout=30)
                    logger.warning("Skipped oversize Firehose frame (%d bytes)", msg_len)
                    return None

                raw = await asyncio.wait_for(self.reader.readexactly(msg_len), timeout=30)
                if raw[:2] == b"\x78\x9c":  # zlib?
                    raw = zlib.decompress(raw)

            payload = json.loads(raw.decode("utf-8"))

            msg_type_id = payload.get("type", 0)
            msg_type = MESSAGE_TYPES.get(msg_type_id, f"unknown_{msg_type_id}")

            ts = payload.get("timestamp", 0)
            if isinstance(ts, str):
                timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)

            msg = FirehoseMessage(
                message_type=msg_type,
                timestamp=timestamp,
                data=payload,
                pitr=payload.get("pitr"),
                raw_type=msg_type_id,
            )
            if msg.ident:
                self._update_cache(msg)
                self._seen_idents.add(msg.ident)
            return msg

        except asyncio.TimeoutError:
            logger.debug("Read timeout; waiting for data/keep‑alive")
            return None
        except asyncio.IncompleteReadError as exc:
            logger.warning("Connection lost mid‑frame: %s", exc)
            self.state = ConnectionState.ERROR
            return None
        except (json.JSONDecodeError, zlib.error) as exc:
            logger.error("Bad Firehose payload: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected read error: %s", exc)
            self.state = ConnectionState.ERROR
            return None


    # ─────────────────────────────────────────────────────────────────────
    def _update_cache(self, message: FirehoseMessage) -> None:
        """Update flight cache with new message."""
        ident = message.ident
        if not ident:
            return
            
        # Clean expired entries periodically
        if len(self.cache) > MAX_CACHE_SIZE:
            self._clean_cache()
            
        # Update or create cache entry
        if ident not in self.cache:
            self.cache[ident] = FlightCache(ident=ident)
            
        self.cache[ident].add_message(message)

    # ─────────────────────────────────────────────────────────────────────
    def _clean_cache(self) -> None:
        """Remove expired cache entries."""
        expired = [k for k, v in self.cache.items() if v.is_expired()]
        for key in expired:
            del self.cache[key]
            
        # If still too large, remove oldest entries
        if len(self.cache) > MAX_CACHE_SIZE:
            sorted_entries = sorted(
                self.cache.items(), 
                key=lambda x: x[1].last_updated
            )
            for key, _ in sorted_entries[:-MAX_CACHE_SIZE]:
                del self.cache[key]

    # ─────────────────────────────────────────────────────────────────────
    async def stream_messages(
        self,
        callback: Callable[[FirehoseMessage], Any],
        duration_seconds: Optional[int] = None,
        target_idents: Optional[Set[str]] = None
    ) -> None:
        """
        Stream messages with optional filtering by ident.
        
        Args:
            callback: Async function to call for each message
            duration_seconds: Max streaming duration
            target_idents: Set of idents to filter for (None = all)
        """
        if self.state != ConnectionState.STREAMING:
            if not await self.connect():
                return

        start = datetime.utcnow()
        message_count = 0
        
        try:
            while self.state == ConnectionState.STREAMING:
                # Check duration limit
                if duration_seconds:
                    elapsed = (datetime.utcnow() - start).total_seconds()
                    if elapsed > duration_seconds:
                        logger.info("Stream duration limit reached")
                        break

                # Read next message
                message = await self.read_message()
                if not message:
                    if self.state == ConnectionState.ERROR:
                        # Try to reconnect
                        logger.info("Attempting reconnection...")
                        if await self.connect():
                            continue
                        else:
                            break
                    continue

                message_count += 1
                
                # Filter by target idents if specified
                if target_idents and message.ident not in target_idents:
                    continue

                # Fire callback
                try:
                    result = await callback(message)
                    # Allow callback to stop streaming
                    if result is False:
                        logger.info("Callback requested stream stop")
                        break
                except Exception as exc:
                    logger.error("Callback error: %s", exc)

        except Exception as exc:
            logger.error("Streaming error: %s", exc)
        finally:
            logger.info(
                "Stream ended - Duration: %.1fs, Messages: %d, Unique flights: %d",
                (datetime.utcnow() - start).total_seconds(),
                message_count,
                len(self._seen_idents)
            )
            await self.disconnect()

    # ─────────────────────────────────────────────────────────────────────
    def get_flight_data(self, ident: str) -> Tuple[List[FirehoseMessage], Optional[Dict]]:
        """
        Get cached flight data and computed summary.
        
        Returns:
            Tuple of (messages, summary)
        """
        ident = ident.upper()
        cache_entry = self.cache.get(ident)
        
        if not cache_entry or cache_entry.is_expired():
            return [], None
            
        # Generate summary if not cached
        if not cache_entry.summary:
            cache_entry.summary = self._summarize_flight(cache_entry.messages)
            
        return cache_entry.messages, cache_entry.summary

    # ─────────────────────────────────────────────────────────────────────
    def _summarize_flight(self, messages: List[FirehoseMessage]) -> Dict[str, Any]:
        """
        Create comprehensive flight summary from messages.
        Enhanced to extract scheduled times and more data.
        """
        summary: Dict[str, Any] = {
            "source": "firehose", 
            "confidence": 0.99,
            "data": {},
            "message_types_seen": list(set(m.message_type for m in messages)),
            "message_count": len(messages)
        }
        
        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp)
        
        # Process each message type
        for msg in messages:
            data = msg.data
            
            if msg.message_type == "flifo":
                # FLIFO has the richest data
                summary["data"].update({
                    "origin": data.get("origin"),
                    "destination": data.get("destination"), 
                    "aircraft_type": data.get("aircrafttype"),
                    "route": data.get("route"),
                    "filed_altitude": data.get("filed_altitude"),
                    "filed_airspeed": data.get("filed_airspeed_kts"),
                })
                
                # Extract scheduled times from FLIFO
                if sched_out := data.get("scheduled_out"):
                    summary["data"]["scheduled_departure"] = sched_out
                    # Convert to HHMM format
                    try:
                        dt = datetime.fromisoformat(sched_out.replace('Z', '+00:00'))
                        summary["data"]["sched_out_local"] = dt.strftime("%H%M")
                    except:
                        pass
                        
                if sched_in := data.get("scheduled_in"):
                    summary["data"]["scheduled_arrival"] = sched_in
                    try:
                        dt = datetime.fromisoformat(sched_in.replace('Z', '+00:00'))
                        summary["data"]["sched_in_local"] = dt.strftime("%H%M")
                    except:
                        pass
                
                # Gate information
                if origin_gate := data.get("origin_gate"):
                    summary["data"]["origin_gate"] = origin_gate
                if dest_gate := data.get("destination_gate"):
                    summary["data"]["destination_gate"] = dest_gate
                    
                # Terminal information
                if origin_terminal := data.get("origin_terminal"):
                    summary["data"]["origin_terminal"] = origin_terminal
                if dest_terminal := data.get("destination_terminal"):
                    summary["data"]["destination_terminal"] = dest_terminal
                    
            elif msg.message_type == "flight_plan":
                # Flight plan has route and altitude info
                summary["data"].update({
                    "route": data.get("route"),
                    "waypoints": data.get("waypoints", []),
                    "filed_altitude": data.get("altitude"),
                    "filed_speed": data.get("speed"),
                })
                
            elif msg.message_type == "departure":
                summary["data"]["actual_departure"] = msg.timestamp.isoformat()
                summary["data"]["departure_delay"] = data.get("delay", 0)
                if runway := data.get("runway"):
                    summary["data"]["departure_runway"] = runway
                    
            elif msg.message_type == "arrival":
                summary["data"]["actual_arrival"] = msg.timestamp.isoformat()
                summary["data"]["arrival_delay"] = data.get("delay", 0)
                if runway := data.get("runway"):
                    summary["data"]["arrival_runway"] = runway
                    
            elif msg.message_type == "cancellation":
                summary["data"]["cancelled"] = True
                summary["data"]["cancellation_time"] = msg.timestamp.isoformat()
                
            elif msg.message_type == "surface_offblock":
                summary["data"]["offblock_time"] = msg.timestamp.isoformat()
                
            elif msg.message_type == "surface_onblock":
                summary["data"]["onblock_time"] = msg.timestamp.isoformat()
                
            elif msg.message_type == "timing_advance":
                # Predictive data
                if eta := data.get("estimated_arrival_time"):
                    summary["data"]["predicted_arrival"] = eta
                if etd := data.get("estimated_departure_time"):
                    summary["data"]["predicted_departure"] = etd
                    
        return summary

# ─────────────────────────────────────────────────────────────────────────────
# ░ ENHANCED VALIDATOR ░
# ─────────────────────────────────────────────────────────────────────────────
class FirehoseValidator:
    """Enhanced validator with better caching and data extraction."""

    def __init__(self) -> None:
        if FIREHOSE_USERNAME and FIREHOSE_PASSWORD:
            self.client = FirehoseClient(FIREHOSE_USERNAME, FIREHOSE_PASSWORD)
        else:
            logger.warning("Firehose credentials not set – disabled")
            self.client = None

    # ─────────────────────────────────────────────────────────────────────
    async def validate_flight(self, flight_no: str, date_str: str) -> Optional[Dict]:
        """
        Validate flight with enhanced caching and schedule support.
        """
        if not self.client:
            return None

        try:
            flight_date = datetime.strptime(date_str, "%m/%d/%Y")
        except ValueError:
            logger.debug("Invalid date format: %s", date_str)
            return None

        ident = flight_no.upper()
        today = datetime.utcnow().date()
        fdate = flight_date.date()
        if fdate < (today - timedelta(days=2)) or fdate > (today + timedelta(days=2)):
            logger.info("Skipping Firehose for %s on %s (outside live window)", ident, date_str)
            return None
        
        # Check cache first
        messages, summary = self.client.get_flight_data(ident)
        if summary:
            logger.info("Using cached Firehose data for %s", ident)
            return summary

        # For recent/current flights, use realtime stream
        days_diff = abs((datetime.now() - flight_date).days)
        if days_diff <= 2:
            return await self._validate_realtime(ident, flight_date)
        else:
            # For future flights, we might still catch schedule updates
            logger.info("Attempting to find schedule data for %s on %s", ident, date_str)
            return await self._validate_schedule(ident, flight_date)

    # ─────────────────────────────────────────────────────────────────────
    async def _validate_realtime(
        self, 
        flight_no: str, 
        flight_date: datetime
    ) -> Optional[Dict]:
        """Enhanced realtime validation with smart streaming."""
        if not self.client:
            return None

        target_ident = flight_no.upper()
        found_messages: List[FirehoseMessage] = []
        stream_duration = 30  # seconds
        
        # Check if we need specific date filtering
        is_today = flight_date.date() == datetime.utcnow().date()
        
        async def collector(msg: FirehoseMessage) -> Optional[bool]:
            """Collect relevant messages."""
            if msg.ident == target_ident:
                # For non-today flights, filter by date
                if not is_today:
                    msg_date = msg.timestamp.date()
                    if msg_date != flight_date.date():
                        return None
                        
                found_messages.append(msg)
                
                # Stop early if we have enough data
                important_types = {"flifo", "departure", "arrival", "cancellation"}
                important_count = sum(
                    1 for m in found_messages 
                    if m.message_type in important_types
                )
                
                if important_count >= 2 or len(found_messages) > 10:
                    return False  # Stop streaming
                    
            return None

        # Stream with target filter for efficiency
        await self.client.stream_messages(
            collector, 
            duration_seconds=stream_duration,
            target_idents={target_ident}
        )

        if not found_messages:
            logger.info("No Firehose data found for %s", target_ident)
            return None

        # Generate and cache summary
        summary = self.client._summarize_flight(found_messages)
        
        # Update cache
        if target_ident in self.client.cache:
            self.client.cache[target_ident].summary = summary
            
        return summary

    # ─────────────────────────────────────────────────────────────────────
    async def _validate_schedule(
        self, 
        flight_no: str,
        flight_date: datetime
    ) -> Optional[Dict]:
        """
        Attempt to find schedule information for future flights.
        This watches for FLIFO messages which often contain schedule data.
        """
        if not self.client:
            return None
            
        # For schedules, we do a shorter stream looking for FLIFO
        target_ident = flight_no.upper()
        found_flifo = None
        
        async def flifo_collector(msg: FirehoseMessage) -> Optional[bool]:
            nonlocal found_flifo
            if msg.ident == target_ident and msg.message_type == "flifo":
                # Check if this FLIFO has schedule data
                if "scheduled_out" in msg.data or "scheduled_in" in msg.data:
                    found_flifo = msg
                    return False  # Stop streaming
            return None
            
        # Quick 10-second scan for FLIFO
        await self.client.stream_messages(
            flifo_collector,
            duration_seconds=10,
            target_idents={target_ident}
        )
        
        if found_flifo:
            # Create summary from single FLIFO
            summary = self.client._summarize_flight([found_flifo])
            summary["confidence"] = 0.85  # Lower confidence for schedule-only
            return summary
            
        return None

# ─────────────────────────────────────────────────────────────────────────────
# ░ PUBLIC INTERFACE ░
# ─────────────────────────────────────────────────────────────────────────────
async def validate_with_firehose(flight: Dict) -> Optional[Dict]:
    """
    Enhanced public interface with better data extraction.
    """
    if not flight.get("flight_no") or not flight.get("date"):
        return None

    validator = FirehoseValidator()
    if not validator.client:
        return None

    try:
        result = await validator.validate_flight(
            flight["flight_no"], 
            flight["date"]
        )
        
        # Log what we found
        if result:
            data_keys = list(result.get("data", {}).keys())
            logger.info(
                "Firehose validation for %s: Found %d fields including %s",
                flight["flight_no"],
                len(data_keys),
                data_keys[:5]  # First 5 fields
            )
            
        return result
        
    except Exception as exc:
        logger.error("Firehose validation error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ░ UTILITY FUNCTIONS ░
# ─────────────────────────────────────────────────────────────────────────────
async def get_firehose_status() -> Dict[str, Any]:
    """Get current Firehose connection status and statistics."""
    if not (FIREHOSE_USERNAME and FIREHOSE_PASSWORD):
        return {"status": "disabled", "reason": "No credentials configured"}
        
    client = FirehoseClient(FIREHOSE_USERNAME, FIREHOSE_PASSWORD)
    
    try:
        # Try to connect
        connected = await client.connect(retry=False)
        
        status = {
            "status": "connected" if connected else "failed",
            "state": client.state.value,
            "cache_size": len(client.cache),
            "unique_flights_seen": len(client._seen_idents),
            "pitr_available": client.pitr_position is not None,
        }
        
        if connected:
            await client.disconnect()
            
        return status
        
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc)
        }