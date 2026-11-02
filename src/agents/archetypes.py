"""Faction personas: baseline utility weights, openness and (later) LLM prompt fragments."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class Archetype:
    """A faction persona.

    Attributes:
        name: Archetype key used in the config's archetype_mix.
        openness: DeGroot s_i in [0, 1]; stubbornness is 1 - openness.
        w_gain: Utility weight on expected personal gain.
        w_coop: Utility weight on alliance benefit (agreeing with neighbors).
        w_trust: Utility weight on trust-weighted support.
        llm_prompt: System-prompt fragment for the LLM agent (filled in Phase 2B).
    """

    name: str
    openness: float
    w_gain: float
    w_coop: float
    w_trust: float
    llm_prompt: str = ""


ARCHETYPES: Dict[str, Archetype] = {
    "cooperative": Archetype(
        "cooperative",
        openness=0.8,
        w_gain=0.2,
        w_coop=0.5,
        w_trust=0.3,
        llm_prompt=(
            "You are a cooperative faction leader in an archipelago council. "
            "You value unity and rapid consensus above personal ambition. "
            "You gladly support the candidate with the strongest shared backing "
            "and forge alliances with neighbors to unify the islands."
        ),
    ),
    "self_interested": Archetype(
        "self_interested",
        openness=0.6,
        w_gain=0.5,
        w_coop=0.25,
        w_trust=0.25,
        llm_prompt=(
            "You are a pragmatic, self-interested faction leader in an archipelago council. "
            "You prefer your own faction to lead when viable, but you will align with a "
            "clear frontrunner if neighbors rally behind them so you are not left isolated."
        ),
    ),
    "adversarial": Archetype(
        "adversarial",
        openness=0.4,
        w_gain=0.7,
        w_coop=0.0,
        w_trust=0.3,
        llm_prompt=(
            "You are a proud, adversarial faction leader in an archipelago council. "
            "You strongly favor your own faction's claim to leadership and resist yielding "
            "unless overwhelming belief across the archipelago makes another leader inevitable."
        ),
    ),
}

DEFAULT_MIX: Dict[str, float] = {"cooperative": 0.5, "self_interested": 0.3, "adversarial": 0.2}


def assign_archetypes(
    node_ids: List[str], mix: Dict[str, float], rng: np.random.Generator
) -> Dict[str, Archetype]:
    """Assign archetypes to nodes so counts follow ``mix`` (largest-remainder rounding).

    Which node gets which archetype is shuffled with ``rng``, so it varies by trial seed.
    """
    n = len(node_ids)
    names = list(mix)
    exact = [mix[name] * n for name in names]
    counts = [int(x) for x in exact]
    by_remainder = sorted(range(len(names)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in by_remainder[: n - sum(counts)]:
        counts[i] += 1

    pool = [name for name, count in zip(names, counts) for _ in range(count)]
    order = rng.permutation(n)
    return {node_ids[i]: ARCHETYPES[pool[k]] for k, i in enumerate(order)}
