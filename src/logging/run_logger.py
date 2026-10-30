"""Logging utilities for event tracking, agent decisions and trial summaries."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]

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

DECISION_FIELDNAMES = [
    "trial_id",
    "round",
    "node",
    "agent_type",
    "valid_json",
    "valid_action",
    "vote",
    "baseline_vote",
    "decision_latency_ms",
    "tokens",
]

SUMMARY_FIELDNAMES = [
    "trial_id",
    "agent_type",
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
    """Logs individual message events, agent decisions and trial-level summaries."""

    def __init__(self, trial_id: str):
        self.trial_id = trial_id
        self.events: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []

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

    def log_decision(
        self,
        round_num: int,
        node: str,
        agent_type: str,
        valid_json: bool,
        valid_action: bool,
        vote: str,
        baseline_vote: Optional[str],
        decision_latency_ms: float,
        tokens: int,
    ) -> None:
        """Record one agent decision. ``baseline_vote`` is what the baseline agent would pick from the same state."""
        self.decisions.append(
            {
                "trial_id": self.trial_id,
                "round": round_num,
                "node": node,
                "agent_type": agent_type,
                "valid_json": valid_json,
                "valid_action": valid_action,
                "vote": vote,
                "baseline_vote": baseline_vote if baseline_vote is not None else "",
                "decision_latency_ms": round(decision_latency_ms, 3),
                "tokens": tokens,
            }
        )

    @staticmethod
    def _write_rows(filepath: PathLike, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_events_csv(self, filepath: PathLike) -> None:
        """Write all recorded events for this trial to a CSV file."""
        self._write_rows(filepath, EVENT_FIELDNAMES, self.events)

    def write_decisions_csv(self, filepath: PathLike) -> None:
        """Write all recorded agent decisions for this trial to a CSV file."""
        self._write_rows(filepath, DECISION_FIELDNAMES, self.decisions)

    @staticmethod
    def append_summary_csv(filepath: PathLike, summary: Dict[str, Any]) -> None:
        """Append a trial summary record to a summary CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists() and path.stat().st_size > 0
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)
