# connections.py
"""
Connection detection logic for Flight-Intel.

Given a list of Flight models, detects reasonable connections between
same-day legs at the same airport within a configurable connection window.
"""

from collections import defaultdict
from typing import Dict, List

import logging
logger = logging.getLogger(__name__)

from models import Flight


class ConnectionDetector:
    @staticmethod
    def find_connections(flights: List[Flight]) -> List[Dict]:
        """
        Detect connections between flights on the same day.

        Rules (same as original implementation):
        - Group by date.
        - Sort each day by sched_out_local (treat None as "0000").
        - A connection exists if:
            * curr.dest == next.origin
            * both have sched_in_local/sched_out_local
            * connection time between 20 and 360 minutes
        - Type:
            * "same_day"  for < 240 min
            * "long_connection" for 240–360 min
        """
        connections: List[Dict] = []
        by_date: Dict[str, List[Flight]] = defaultdict(list)

        # Group flights by date
        for f in flights:
            if f.date:
                by_date[f.date].append(f)

        # Process each day separately
        for date, day_flights in by_date.items():
            # Sort by departure time (None → "0000")
            day_flights.sort(key=lambda x: x.sched_out_local or "0000")

            for i in range(len(day_flights) - 1):
                curr = day_flights[i]
                nxt = day_flights[i + 1]

                if (
                    curr.dest
                    and nxt.origin
                    and curr.dest == nxt.origin
                    and curr.sched_in_local
                    and nxt.sched_out_local
                ):
                    try:
                        # minutes since midnight
                        arr = (
                            int(curr.sched_in_local[:2]) * 60
                            + int(curr.sched_in_local[2:])
                        )
                        dep = (
                            int(nxt.sched_out_local[:2]) * 60
                            + int(nxt.sched_out_local[2:])
                        )

                        # Handle overnight crossover
                        if dep < arr:
                            dep += 24 * 60

                        conn = dep - arr

                        # 20–360 min window
                        if 20 <= conn <= 360:
                            connections.append(
                                {
                                    "from_flight": curr.flight_no,
                                    "to_flight": nxt.flight_no,
                                    "at_airport": curr.dest,
                                    "connection_time": conn,  # minutes
                                    "type": "same_day" if conn < 240 else "long_connection",
                                    "date": date,
                                }
                            )
                    except Exception:
                        # Ignore parsing errors and keep scanning
                        continue

        logger.info(f"Detected {len(connections)} connections in schedule")
        return connections
