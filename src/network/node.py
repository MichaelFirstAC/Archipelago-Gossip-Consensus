"""Node class representing a faction peer in the gossip network."""

import asyncio
from typing import Any, Dict, List, Optional


class Node:
    """A peer node in the gossip consensus simulation.

    Attributes:
        id: Unique identifier for the faction node (e.g., 'faction_0').
        faction_name: Human-readable display name.
        neighbors: List of peer node IDs adjacent to this node.
        inbox: Asynchronous queue for incoming messages.
        belief: Current belief/allegiance state (e.g., designated leader or allied faction).
        agent: The decision agent (DummyAgent or LLMAgent) governing this node.
    """

    def __init__(
        self,
        node_id: str,
        faction_name: Optional[str] = None,
        neighbors: Optional[List[str]] = None,
        initial_belief: Optional[str] = None,
        agent: Any = None,
    ):
        self.id = node_id
        self.faction_name = faction_name or node_id.replace("_", " ").title()
        self.neighbors: List[str] = list(neighbors) if neighbors is not None else []
        self.belief: str = initial_belief if initial_belief is not None else node_id
        self.agent: Any = agent
        self.inbox: asyncio.Queue = asyncio.Queue()

    async def receive(self, message: Dict[str, Any]) -> None:
        """Enqueue an incoming message into the node's inbox."""
        await self.inbox.put(message)

    def drain_inbox(self) -> List[Dict[str, Any]]:
        """Extract and clear all messages currently queued in the inbox."""
        messages: List[Dict[str, Any]] = []
        while not self.inbox.empty():
            try:
                messages.append(self.inbox.get_nowait())
                self.inbox.task_done()
            except asyncio.QueueEmpty:
                break
        return messages

    def get_state(self) -> Dict[str, Any]:
        """Return the current public state of the node."""
        return {
            "node_id": self.id,
            "faction_name": self.faction_name,
            "belief": self.belief,
        }

    def reset(self, initial_belief: Optional[str] = None) -> None:
        """Reset inbox and belief state for a fresh trial run."""
        self.belief = initial_belief if initial_belief is not None else self.id
        self.inbox = asyncio.Queue()
