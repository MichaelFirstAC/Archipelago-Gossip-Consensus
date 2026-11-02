"""Builds the agent type named by `agent_type` in the experiment config."""

from typing import List, Optional

from src.agents.archetypes import Archetype
from src.agents.base_agent import BaseAgent
from src.agents.baseline_agent import BaselineAgent
from src.agents.dummy_agent import DummyAgent
from src.agents.llm_agent import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    LLMAgent,
)

AGENT_TYPES = ("dummy", "baseline", "llm")


def create_agent(
    agent_type: str,
    node_id: str,
    candidates: List[str],
    archetype: Archetype,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = DEFAULT_SEED,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> BaseAgent:
    """Return one agent for one node: 'dummy', 'baseline' or 'llm'."""
    if agent_type == "dummy":
        return DummyAgent(mode="scripted", node_id=node_id)
    if agent_type == "baseline":
        return BaselineAgent(node_id, candidates, archetype)
    if agent_type == "llm":
        return LLMAgent(
            node_id=node_id,
            candidates=candidates,
            archetype=archetype,
            model=model,
            temperature=temperature,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unknown agent_type: {agent_type!r}. Expected one of {AGENT_TYPES}.")
