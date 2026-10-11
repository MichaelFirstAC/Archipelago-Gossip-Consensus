"""Consensus evaluation logic for the Archipelago simulation."""

from typing import List, Optional, Tuple
from src.network.node import Node


def check_consensus(nodes: List[Node]) -> Tuple[bool, Optional[str]]:
    """Determine whether all nodes have converged on the same belief state.

    Args:
        nodes: List of Node instances participating in the simulation.

    Returns:
        A tuple of (converged: bool, consensus_value: Optional[str]).
        If all nodes hold the exact same non-empty belief, converged is True and
        consensus_value is that shared belief. Otherwise (False, None).
    """
    if not nodes:
        return False, None

    target = nodes[0].belief
    if target is None:
        return False, None

    for node in nodes[1:]:
        if node.belief != target:
            return False, None

    return True, target


def is_converged(nodes: List[Node]) -> bool:
    """Convenience boolean check for convergence."""
    converged, _ = check_consensus(nodes)
    return converged
