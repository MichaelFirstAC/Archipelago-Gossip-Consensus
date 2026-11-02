"""LLM-driven faction agent using a local open-weight model via LiteLLM."""

import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from src.agents import belief as bel
from src.agents.archetypes import ARCHETYPES, Archetype
from src.agents.base_agent import BaseAgent
from src.agents.schema import Action

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_litellm_module: Any = None


def _get_litellm() -> Any:
    """Lazy-load litellm once per process to avoid import overhead when not calling a real model."""
    global _litellm_module
    if _litellm_module is None:
        import litellm as _ll

        _ll.suppress_debug_info = True
        _litellm_module = _ll
    return _litellm_module

DEFAULT_MODEL = "ollama/llama3.2:3b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 64

# Process-level registry of models already initialized/warmed in this process.
_LOADED_MODELS: Set[str] = set()

SCHEMA_INSTRUCTION = (
    "Respond ONLY with a single compact JSON object (no markdown, no explanation). "
    'Required keys: "action" (one of "forge_alliance", "break_alliance", "hold", "broadcast_state"), '
    '"target" (a faction_id string or null), '
    '"vote" (the faction_id you vote for as leader), '
    '"dialogue" (one short first-person sentence under 12 words).\n'
    'Example: {"action":"forge_alliance","target":"faction_1","vote":"faction_1","dialogue":"I support faction_1 to unite our islands."}'
)

REPAIR_PROMPT = (
    "Your last output was invalid JSON or did not match the exact schema. "
    "Respond again ONLY with a valid JSON object containing exactly keys "
    '"action" ("forge_alliance", "break_alliance", "hold", or "broadcast_state"), '
    '"target" (a faction_id string or null), '
    '"vote" (a valid candidate faction_id string), and '
    '"dialogue" (a short first-person string).'
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def normalize_model_name(model: str) -> str:
    """Ensure bare Ollama model tags carry the 'ollama/' prefix for LiteLLM."""
    if "/" not in model:
        return f"ollama/{model}"
    return model


def _clean_json_text(raw: str) -> str:
    """Strip markdown code fences or surrounding whitespace before JSON validation."""
    text = raw.strip()
    if text.startswith("```"):
        text = _CODE_FENCE_RE.sub("", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text


class LLMAgent(BaseAgent):
    """Wraps a local open-weight LLM via LiteLLM with strict Pydantic validation and 1-retry repair.

    Attributes:
        node_id: Unique faction identifier (e.g. 'faction_0').
        candidates: Ordered list of candidate faction IDs.
        archetype: Faction persona containing system-prompt fragment and openness.
        model: LiteLLM model identifier (fixed across the experiment).
        temperature: Sampling temperature.
        seed: Random seed passed to the model for reproducibility.
        max_new_tokens: Cap on generated tokens per call for efficiency.
        last_tokens: Total tokens consumed in the most recent `decide()` call.
        last_latency_ms: Wall-clock milliseconds spent in the most recent `decide()` call.
        last_valid_json: True iff the first LLM attempt in `decide()` parsed against `Action`.
        last_parse_failed: True iff both the initial attempt and the repair retry failed.
        parse_failures: Cumulative count of double-failures for this agent instance.
    """

    def __init__(
        self,
        node_id: str,
        candidates: Optional[List[str]] = None,
        archetype: Optional[Archetype] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: Optional[int] = DEFAULT_SEED,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        completion_fn: Optional[Callable[..., Any]] = None,
    ):
        self.node_id = node_id
        self.candidates = list(candidates) if candidates is not None else [node_id]
        self.archetype = archetype or ARCHETYPES["cooperative"]
        self.model = normalize_model_name(model)
        self.temperature = float(temperature)
        self.seed = seed
        self.max_new_tokens = int(max_new_tokens)
        self._completion_fn = completion_fn

        # Per-decision telemetry read by GossipProtocol / RunLogger
        self.last_tokens: int = 0
        self.last_latency_ms: float = 0.0
        self.last_valid_json: bool = True
        self.last_parse_failed: bool = False
        self.parse_failures: int = 0

        _LOADED_MODELS.add(self.model)

    def build_messages(self, state: Dict[str, Any], inbox: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Build a concise first-person prompt from archetype, current state, and inbox gossip."""
        node_id = state.get("node_id", self.node_id)
        current_vote = state.get("vote", node_id)
        candidates = state.get("candidates") or self.candidates
        neighbors = state.get("neighbors") or []

        belief_vec = state.get("belief_vector")
        if belief_vec is not None:
            arr = np.asarray(belief_vec, dtype=float)
            top_idx = bel.top_candidate(arr)
            top_id = candidates[top_idx]
            top_val = float(arr[top_idx])
            scores_str = ", ".join(f"{cid}:{arr[i]:.2f}" for i, cid in enumerate(candidates))
        else:
            top_id = state.get("belief", current_vote)
            top_val = 1.0
            scores_str = f"{top_id}:1.00"

        if inbox:
            gossip_items = []
            for msg in inbox:
                sender = msg.get("sender", "unknown")
                payload = msg.get("payload", {})
                s_vec = payload.get("belief_vector")
                if s_vec is not None:
                    s_top = candidates[bel.top_candidate(np.asarray(s_vec, dtype=float))]
                else:
                    s_top = payload.get("belief", "unknown")
                gossip_items.append(f"{sender} backs {s_top}")
            inbox_str = "; ".join(gossip_items)
        else:
            inbox_str = "none"

        persona = self.archetype.llm_prompt or f"You are a {self.archetype.name} faction leader."
        system_content = f"{persona}\n\n{SCHEMA_INSTRUCTION}"

        user_content = (
            f"I am {node_id}. Valid candidates: {', '.join(candidates)}. "
            f"Neighbors: {', '.join(neighbors) if neighbors else 'none'}.\n"
            f"My previous vote: {current_vote}. "
            f"Current belief scores: [{scores_str}] (leading candidate: {top_id} with score {top_val:.2f}).\n"
            f"Incoming gossip this round: {inbox_str}.\n"
            f"Choose my next action and vote based on the belief scores above "
            f"(vote MUST be one of {candidates}; target MUST be null or a valid faction id)."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _call_model(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """Invoke LiteLLM once and return (raw_content, total_tokens)."""
        litellm_model = (
            "ollama_chat/" + self.model.split("/", 1)[1]
            if self.model.startswith("ollama/")
            else self.model
        )
        kwargs: Dict[str, Any] = {
            "model": litellm_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if litellm_model.startswith("ollama"):
            api_base = os.environ.get("OLLAMA_API_BASE") or os.environ.get("OLLAMA_HOST")
            if api_base:
                if not api_base.startswith("http"):
                    api_base = f"http://{api_base}"
                kwargs["api_base"] = api_base
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": Action.model_json_schema()},
            }
            kwargs["num_ctx"] = 512
            kwargs["keep_alive"] = "30m"

        fn = self._completion_fn or _get_litellm().completion
        response = fn(**kwargs)

        # Extract text content
        choice = response.choices[0]
        content = getattr(choice.message, "content", None) or ""

        # Extract token usage
        usage = getattr(response, "usage", None)
        tokens = 0
        if usage is not None:
            tokens = int(getattr(usage, "total_tokens", 0) or 0)
            if tokens == 0:
                prompt_t = int(getattr(usage, "prompt_tokens", 0) or 0)
                comp_t = int(getattr(usage, "completion_tokens", 0) or 0)
                tokens = prompt_t + comp_t
        if tokens == 0 and content:
            # Fallback estimate if backend omitted usage metadata
            tokens = max(1, len(content.split()))

        return content, tokens

    @staticmethod
    def _parse_action(raw_text: str) -> Action:
        """Validate raw LLM output against the strict Pydantic Action schema."""
        cleaned = _clean_json_text(raw_text)
        return Action.model_validate_json(cleaned)

    def decide(self, state: Dict[str, Any], inbox: List[Dict[str, Any]]) -> Action:
        """Choose an Action for this round with a retry-once-then-hold policy."""
        started = time.perf_counter()
        self.last_tokens = 0
        self.last_valid_json = False
        self.last_parse_failed = False

        messages = self.build_messages(state, inbox)
        raw_first = ""

        # Attempt 1
        try:
            raw_first, tokens_1 = self._call_model(messages)
            self.last_tokens += tokens_1
            action = self._parse_action(raw_first)
            self.last_valid_json = True
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            return action
        except Exception:
            pass

        # Attempt 2: one retry with repair prompt
        repair_messages = list(messages) + [
            {"role": "assistant", "content": raw_first or "<invalid>"},
            {"role": "user", "content": REPAIR_PROMPT},
        ]
        try:
            raw_second, tokens_2 = self._call_model(repair_messages)
            self.last_tokens += tokens_2
            action = self._parse_action(raw_second)
            self.last_valid_json = False
            self.last_parse_failed = False
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            return action
        except Exception:
            # Second failure: log parse-failure and hold previous state
            self.last_valid_json = False
            self.last_parse_failed = True
            self.parse_failures += 1
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            fallback_vote = state.get("vote", self.node_id)
            return Action(
                action="hold",
                target=None,
                vote=fallback_vote,
                dialogue=f"{self.node_id} holds previous position after parse failure.",
            )
