"""Scripted and random dummy agent for testing network layer independently."""

import random
from typing import Any, Dict, List, Optional


VALID_ACTIONS = ["forge_alliance", "break_alliance", "hold", "broadcast_state"]


class DummyAgent:
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

    def decide(
        self,
        node_id: str,
        current_belief: str,
        inbox_messages: List[Dict[str, Any]],
        known_neighbors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate an action decision based on incoming gossip messages.

        Args:
            node_id: The ID of the current node.
            current_belief: The current faction/leader belief of this node.
            inbox_messages: List of message dictionaries delivered during this round.
            known_neighbors: List of neighboring node IDs.

        Returns:
            Dictionary matching the agent action schema:
            {'action': str, 'target': Optional[str], 'dialogue': str}
        """
        if self.mode == "random":
            action = random.choice(VALID_ACTIONS)
            target = random.choice(known_neighbors) if known_neighbors and action in ("forge_alliance", "break_alliance", "broadcast_state") else None
            return {
                "action": action,
                "target": target,
                "dialogue": f"Random decision by {node_id}.",
            }

        # Scripted consensus-seeking behavior
        if not inbox_messages:
            return {
                "action": "hold",
                "target": None,
                "dialogue": f"{node_id} received no messages; holding current allegiance.",
            }

        # Inspect beliefs reported by neighbors
        candidates = [current_belief]
        for msg in inbox_messages:
            payload = msg.get("payload", {})
            belief = payload.get("belief") or msg.get("belief")
            if belief:
                candidates.append(belief)

        # Consensus target: prioritize minimum lexicographical candidate (e.g. faction_0)
        best_candidate = min(candidates)

        if best_candidate != current_belief:
            return {
                "action": "forge_alliance",
                "target": best_candidate,
                "dialogue": f"{node_id} accepts alliance with {best_candidate}.",
            }

        return {
            "action": "broadcast_state",
            "target": current_belief,
            "dialogue": f"{node_id} continues supporting {current_belief}.",
        }
