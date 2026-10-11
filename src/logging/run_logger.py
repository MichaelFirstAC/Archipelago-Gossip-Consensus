"""Logging utilities for event tracking and trial summaries."""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


EVENT_FIELDNAMES = [
    "trial_id",
    "round",
    "sender",
    "receiver",
    "sent_at",
    "latency_ms",
    "dropped",
    "delivered_at",
]

SUMMARY_FIELDNAMES = [
    "trial_id",
    "latency",
    "loss_rate",
    "topology",
    "archetype_mix",
    "converged",
    "rounds_to_consensus",
    "total_messages",
    "parse_failures",
]


class RunLogger:
    """Logs individual message events and trial-level summary outcomes."""

    def __init__(self, trial_id: str):
        self.trial_id = trial_id
        self.events: List[Dict[str, Any]] = []

    def log_event(
        self,
        round_num: int,
        sender: str,
        receiver: str,
        sent_at: float,
        latency_ms: float,
        dropped: bool,
        delivered_at: Optional[float] = None,
    ) -> None:
        """Record a single message transmission event."""
        self.events.append(
            {
                "trial_id": self.trial_id,
                "round": round_num,
                "sender": sender,
                "receiver": receiver,
                "sent_at": round(sent_at, 4),
                "latency_ms": latency_ms,
                "dropped": dropped,
                "delivered_at": round(delivered_at, 4) if delivered_at is not None else "",
            }
        )

    def write_events_csv(self, filepath: UnionPath) -> None:
        """Write all recorded events for this trial to a CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.events)

    @staticmethod
    def append_summary_csv(filepath: Any, summary: Dict[str, Any]) -> None:
        """Append a trial summary record to a summary CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists() and path.stat().st_size > 0
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)


UnionPath = Any
