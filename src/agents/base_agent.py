"""Shared agent interface so the network never knows which agent type it runs."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.agents.schema import Action


class BaseAgent(ABC):
    """Common interface implemented by dummy, baseline and LLM agents."""

    @abstractmethod
    def decide(self, state: Dict[str, Any], inbox: List[Dict[str, Any]]) -> Action:
        """Choose an action for one round.

        Args:
            state: The node's public state plus its neighbors, with keys
                'node_id', 'faction_name', 'belief', 'vote', 'neighbors'.
            inbox: Messages delivered to the node this round.

        Returns:
            A validated Action.
        """
