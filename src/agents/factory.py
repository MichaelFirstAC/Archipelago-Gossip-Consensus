"""Builds the agent type named by `agent_type` in the experiment config."""

from typing import List

from src.agents.archetypes import Archetype
from src.agents.base_agent import BaseAgent
from src.agents.baseline_agent import BaselineAgent
from src.agents.dummy_agent import DummyAgent

AGENT_TYPES = ("dummy", "baseline", "llm")


def create_agent(agent_type: str, node_id: str, candidates: List[str], archetype: Archetype) -> BaseAgent:
    """Return one agent for one node: 'dummy', 'baseline' or 'llm' (added in Phase 2B)."""
    if agent_type == "dummy":
        return DummyAgent(mode="scripted", node_id=node_id)
    if agent_type == "baseline":
        return BaselineAgent(node_id, candidates, archetype)
    if agent_type == "llm":
        raise NotImplementedError("The LLM agent is built in Phase 2B.")
    raise ValueError(f"Unknown agent_type: {agent_type!r}. Expected one of {AGENT_TYPES}.")
