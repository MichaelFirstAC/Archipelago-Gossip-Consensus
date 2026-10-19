"""Scripted and random dummy agent for testing network layer independently."""

import random
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.agents.schema import Action


VALID_ACTIONS = ["forge_alliance", "break_alliance", "hold", "broadcast_state"]


class DummyAgent(BaseAgent):
    """Scripted dummy agent providing immediate valid action decisions without LLM calls.

    Attributes:
        mode: 'scripted' (default) for deterministic consensus propagation,
              or 'random' for randomized action sampling.
        node_id: Optional ID of the node this agent is bound to.
    """

    def __init__(self, mode: str = "scripted", node_id: Optional[str] = None):
        if mode not in ("scripted", "random"):
            raise ValueError(f"Unknown dummy agent mode: {mode}")
        self.mode = mode
        self.node_id = node_id

    def decide(self, state: Dict[str, Any], inbox: List[Dict[str, Any]]) -> Action:
        """Generate an action decision based on incoming gossip messages.

        Args:
            state: Node state with 'node_id', 'belief', 'vote' and 'neighbors'.
            inbox: List of message dictionaries delivered during this round.

        Returns:
            An Action matching the agent action schema.
        """
        node_id = state["node_id"]
        current_belief = state["belief"]
        current_vote = state.get("vote", current_belief)
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
                vote=current_vote,
                dialogue=f"Random decision by {node_id}.",
            )

        # Scripted consensus-seeking behavior
        if not inbox:
            return Action(
                action="hold",
                target=None,
                vote=current_vote,
                dialogue=f"{node_id} received no messages; holding current allegiance.",
            )

        # Inspect beliefs reported by neighbors
        candidates = [current_belief]
        for msg in inbox:
            payload = msg.get("payload", {})
            belief = payload.get("belief") or msg.get("belief")
            if belief:
                candidates.append(belief)

        # Consensus target: prioritize minimum lexicographical candidate (e.g. faction_0)
        best_candidate = min(candidates)

        if best_candidate != current_belief:
            return Action(
                action="forge_alliance",
                target=best_candidate,
                vote=best_candidate,
                dialogue=f"{node_id} accepts alliance with {best_candidate}.",
            )

        return Action(
            action="broadcast_state",
            target=current_belief,
            vote=current_belief,
            dialogue=f"{node_id} continues supporting {current_belief}.",
        )
