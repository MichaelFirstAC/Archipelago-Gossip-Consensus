"""Simulated network link injecting latency and packet loss."""

import asyncio
import random
import time
from typing import Any, Dict, Optional

from src.logging.run_logger import RunLogger
from src.network.node import Node


class Link:
    """Wraps message delivery across the network with configurable delay and loss.

    Attributes:
        latency_ms: Milliseconds of latency to sleep before delivery.
        loss_rate: Probability [0.0, 1.0] that a transmitted packet is dropped.
        logger: Optional RunLogger instance to record transmission events.
    """

    def __init__(
        self,
        latency_ms: float = 0.0,
        loss_rate: float = 0.0,
        logger: Optional[RunLogger] = None,
    ):
        self.latency_ms = float(latency_ms)
        self.loss_rate = float(loss_rate)
        self.logger = logger

    async def send(
        self,
        round_num: int,
        sender_id: str,
        receiver_node: Node,
        message: Dict[str, Any],
    ) -> bool:
        """Transmit a message asynchronously to a recipient node.

        Simulates network latency and probabilistic loss.
        Emits a log event regardless of whether the message was delivered or dropped.

        Args:
            round_num: Current simulation round.
            sender_id: Identifier of transmitting node.
            receiver_node: Recipient Node object.
            message: Dictionary payload.

        Returns:
            True if the packet was delivered, False if dropped.
        """
        sent_at = time.time()

        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

        # Drop determination
        dropped = random.random() < self.loss_rate

        delivered_at = None
        if not dropped:
            delivered_at = time.time()
            await receiver_node.receive(message)

        if self.logger is not None:
            self.logger.log_event(
                round_num=round_num,
                sender=sender_id,
                receiver=receiver_node.id,
                sent_at=sent_at,
                latency_ms=self.latency_ms,
                dropped=dropped,
                delivered_at=delivered_at,
            )

        return not dropped
