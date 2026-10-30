"""Utility-based, non-LLM baseline agent."""

from typing import Any, Dict, List

import numpy as np

from src.agents import belief as bel
from src.agents.archetypes import Archetype
from src.agents.base_agent import BaseAgent
from src.agents.schema import Action

# Payoff multiplier for leader candidates: leading yourself pays more than another faction leading.
SELF_GAIN = 1.0
OTHER_GAIN = 0.97


class BaselineAgent(BaseAgent):
    """Picks the vote maximizing U = w_gain*gain + w_coop*alliance + w_trust*trust.

    The agent keeps no hidden state: everything comes from ``state`` (including the
    belief vector maintained by the gossip loop) and the inbox, so it is deterministic
    and can also be queried as a reference ("baseline_vote") for any other agent.
    """

    def __init__(self, node_id: str, candidates: List[str], archetype: Archetype):
        self.node_id = node_id
        self.candidates = list(candidates)
        self.archetype = archetype
        self._payoff = np.array([SELF_GAIN if c == node_id else OTHER_GAIN for c in self.candidates])

    def utilities(self, state: Dict[str, Any], inbox: List[Dict[str, Any]]) -> np.ndarray:
        """Utility of voting for each candidate, in ``candidates`` order."""
        own = np.asarray(state["belief_vector"], dtype=float)
        received = [
            (m["sender"], np.asarray(m["payload"]["belief_vector"], dtype=float))
            for m in inbox
            if "belief_vector" in m.get("payload", {})
        ]

        expected_gain = own * self._payoff
        if received:
            # share of arrived messages whose sender currently favors each candidate
            alliance = np.zeros(len(self.candidates))
            for _, vec in received:
                alliance[bel.top_candidate(vec)] += 1.0 / len(received)
            trust = bel.trust_weighted_mean(received)
        else:
            alliance = np.zeros(len(self.candidates))
            alliance[self.candidates.index(state["vote"])] = 1.0
            trust = own

        a = self.archetype
        return a.w_gain * expected_gain + a.w_coop * alliance + a.w_trust * trust

    def decide(self, state: Dict[str, Any], inbox: List[Dict[str, Any]]) -> Action:
        vote = self.candidates[bel.top_candidate(self.utilities(state, inbox))]
        previous = state["vote"]

        if vote != previous:
            if vote == self.node_id:
                action, target = "break_alliance", previous
            else:
                action, target = "forge_alliance", vote
            dialogue = f"{self.node_id} now backs {vote}."
        elif inbox:
            action, target = "broadcast_state", None
            dialogue = f"{self.node_id} keeps backing {vote}."
        else:
            action, target = "hold", None
            dialogue = f"{self.node_id} hears nothing and holds."

        return Action(action=action, target=target, vote=vote, dialogue=dialogue)
