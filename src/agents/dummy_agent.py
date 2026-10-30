"""Scripted and random dummy agent for testing network layer independently."""

import random
from typing import Any, Dict, List, Optional

from src.agents import belief as bel
from src.agents.base_agent import BaseAgent
from src.agents.schema import Action


VALID_ACTIONS = ["forge_alliance", "break_alliance", "hold", "broadcast_state"]


class DummyAgent(BaseAgent):
    """Scripted dummy agent providing immediate valid action decisions without LLM calls.

    Attributes:
        mode: 'scripted' (default) always votes for the candidate its belief vector
              favors; 'random' samples actions and votes at random.
        node_id: Optional ID of the node this agent is bound to.
    """

    def __init__(self, mode: str = "scripted", node_id: Optional[str] = None):
        if mode not in ("scripted", "random"):
            raise ValueError(f"Unknown dummy agent mode: {mode}")
        self.mode = mode
        self.node_id = node_id

    def decide(self, state: Dict[str, Any], inbox: List[Dict[str, Any]]) -> Action:
        """Generate an action decision from the node state and this round's inbox.

        Args:
            state: Node state with 'node_id', 'vote', 'belief_vector', 'candidates', 'neighbors'.
            inbox: List of message dictionaries delivered during this round.

        Returns:
            An Action matching the agent action schema.
        """
        node_id = state["node_id"]
        candidates = state["candidates"]
        neighbors = state.get("neighbors") or []

        if self.mode == "random":
            action = random.choice(VALID_ACTIONS)
            target = (
                random.choice(neighbors)
                if neighbors and action in ("forge_alliance", "break_alliance", "broadcast_state")
                else None
            )
            return Action(
                action=action,
                target=target,
                vote=random.choice(candidates),
                dialogue=f"Random decision by {node_id}.",
            )

        vote = candidates[bel.top_candidate(state["belief_vector"])]
        if not inbox:
            return Action(
                action="hold",
                target=None,
                vote=vote,
                dialogue=f"{node_id} received no messages; holding.",
            )
        return Action(
            action="broadcast_state",
            target=None,
            vote=vote,
            dialogue=f"{node_id} supports {vote}.",
        )
