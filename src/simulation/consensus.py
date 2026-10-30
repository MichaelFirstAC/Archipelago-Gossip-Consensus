"""Consensus evaluation logic for the Archipelago simulation."""

from typing import List, Optional, Tuple

from src.agents.belief import consensus_score
from src.network.node import Node

DEFAULT_EPSILON = 0.05


def check_consensus(nodes: List[Node], epsilon: float = DEFAULT_EPSILON) -> Tuple[bool, Optional[str]]:
    """Determine whether all nodes have converged on the same outcome.

    Consensus means every node's vote is the same AND the maximum pairwise
    total-variation distance between belief vectors is below ``epsilon``.

    Returns:
        (converged, consensus_value). consensus_value is the shared vote when
        converged, otherwise None.
    """
    if not nodes or any(node.belief_vector is None for node in nodes):
        return False, None

    votes = {node.vote for node in nodes}
    if len(votes) != 1:
        return False, None

    if consensus_score([node.belief_vector for node in nodes]) >= epsilon:
        return False, None

    return True, votes.pop()


def is_converged(nodes: List[Node], epsilon: float = DEFAULT_EPSILON) -> bool:
    """Convenience boolean check for convergence."""
    converged, _ = check_consensus(nodes, epsilon)
    return converged
