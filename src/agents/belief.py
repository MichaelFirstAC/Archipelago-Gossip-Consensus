"""Belief vectors and DeGroot-style belief dynamics (our own math component).

Each node keeps a probability vector over the K candidate outcomes (one per
faction, i.e. the preferred unifying leader). This module is pure numpy so it
can be unit-tested by hand and shared by the baseline agent, the gossip loop
and the consensus check.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# (sender id, sender's belief vector)
Received = Tuple[str, np.ndarray]


def initial_belief(k: int, own_index: int, rng: np.random.Generator, own_bias: float = 0.25) -> np.ndarray:
    """Draw a random starting belief that leans toward the node's own faction.

    The random Dirichlet part breaks symmetry between trials (so the averaged
    consensus belief has a unique leader); ``own_bias`` models faction self-interest.
    """
    belief = (1.0 - own_bias) * rng.dirichlet(np.ones(k))
    belief[own_index] += own_bias
    return belief


def trust_weighted_mean(received: Sequence[Received], trust: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Return sum_j T_ij * b_j, with T_ij the normalized trust over senders that arrived.

    Without an explicit ``trust`` map every arrived message is weighted equally.
    """
    weights = np.array([1.0 if trust is None else trust.get(sender, 1.0) for sender, _ in received])
    weights = weights / weights.sum()
    return sum(w * vec for w, (_, vec) in zip(weights, received))


def degroot_update(
    own: np.ndarray,
    received: Sequence[Received],
    openness: float,
    trust: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """One DeGroot-style step: b_i(t+1) = (1 - s_i) * b_i(t) + s_i * sum_j T_ij * b_j(t).

    ``openness`` is s_i (stubbornness = 1 - s_i). Messages dropped by the network
    are simply absent from ``received``; with no messages the belief is unchanged.
    """
    if not received:
        return own.copy()
    return (1.0 - openness) * own + openness * trust_weighted_mean(received, trust)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance between two distributions (0 = identical, 1 = disjoint)."""
    return 0.5 * float(np.abs(p - q).sum())


def consensus_score(beliefs: Sequence[np.ndarray]) -> float:
    """Maximum pairwise total-variation distance between nodes' beliefs (0 = identical)."""
    score = 0.0
    for i in range(len(beliefs)):
        for j in range(i + 1, len(beliefs)):
            score = max(score, total_variation(beliefs[i], beliefs[j]))
    return score


def top_candidate(belief: np.ndarray) -> int:
    """Index of the most-believed candidate (ties go to the lowest index)."""
    return int(np.argmax(belief))
