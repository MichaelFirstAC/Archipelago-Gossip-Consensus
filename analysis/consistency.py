"""Decision consistency: how often the LLM repeats its vote when asked the same question again.

SPEC section 5.5 defines decision consistency as the share of identical votes when
the same state is re-queried several times. The trial logs cannot answer this (every
state occurred once), so this script runs a small extra experiment:

1. Replay a few cheap baseline trials and record the (state, inbox) every agent faced,
   so the situations are real gossip states, not invented ones.
2. Sample situations evenly across the three archetypes.
3. Ask the LLM each situation several times, with a different model seed per repeat.
   (The same seed would mostly replay the same sample, which would hide the randomness
   the agent shows across trials.) Model, temperature and prompt are the experiment's own.
4. Per situation, consistency is the share of valid votes that match its most common vote.

Needs Ollama running with the experiment's model. From the repo root:
    python -m analysis.consistency [--situations-per-archetype 14] [--repeats 5]

Outputs in ``--out-dir``: ``decision_consistency.csv`` (summary) and
``decision_consistency_raw.csv`` (one row per query).
"""

import argparse
import asyncio
import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest import mock

import numpy as np
import pandas as pd

from src.agents.archetypes import ARCHETYPES, Archetype
from src.agents.base_agent import BaseAgent
from src.agents.baseline_agent import BaselineAgent
from src.agents.llm_agent import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE, LLMAgent
from src.agents.schema import Action
from src.simulation.monte_carlo import load_experiment_config
from src.simulation.runner import run_trial

DEFAULT_OUT_DIR = "results/tables"
# (latency ms, loss rate) of the baseline trials replayed to collect states: a clean
# network, a lossy one, and a slow one, so inboxes range from full to empty and stale.
COLLECT_CONDITIONS = [(0.0, 0.0), (0.0, 0.3), (2000.0, 0.15)]
COLLECT_TRIALS_PER_CONDITION = 4
COLLECT_TIME_SCALE = 0.02  # same speed-up the Monte Carlo sweep uses


class RecordingAgent(BaseAgent):
    """Baseline agent that also stores every (state, inbox) it is asked to decide on."""

    def __init__(self, node_id: str, candidates: List[str], archetype: Archetype, records: List[Dict[str, Any]]):
        self._inner = BaselineAgent(node_id, candidates, archetype)
        self._archetype = archetype
        self._records = records
        self._calls = 0

    def decide(self, state: Dict[str, Any], inbox: List[Dict[str, Any]]) -> Action:
        self._calls += 1
        self._records.append(
            {
                "archetype": self._archetype.name,
                "round": self._calls,
                "state": copy.deepcopy(state),
                "inbox": copy.deepcopy(inbox),
            }
        )
        return self._inner.decide(state, inbox)


async def collect_states(num_agents: int, max_rounds: int, fanout: int, round_ms: float, base_seed: int) -> List[Dict[str, Any]]:
    """Run baseline trials and return every state/inbox the agents decided on."""
    records: List[Dict[str, Any]] = []

    def make_recorder(agent_type, node_id, candidates, archetype, **_):
        return RecordingAgent(node_id, candidates, archetype, records)

    trial = 0
    with mock.patch("src.simulation.runner.create_agent", side_effect=make_recorder):
        for latency, loss in COLLECT_CONDITIONS:
            for _ in range(COLLECT_TRIALS_PER_CONDITION):
                await run_trial(
                    trial_id=f"consistency_collect_{trial:03d}",
                    latency_ms=latency,
                    loss_rate=loss,
                    num_agents=num_agents,
                    max_rounds=max_rounds,
                    fanout=fanout,
                    round_ms=round_ms,
                    agent_type="baseline",
                    seed=base_seed + trial,
                    save_logs=False,
                    time_scale=COLLECT_TIME_SCALE,
                )
                trial += 1
    return records


def sample_situations(records: List[Dict[str, Any]], per_archetype: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Pick ``per_archetype`` recorded situations at random for each archetype."""
    situations = []
    for name in ARCHETYPES:
        pool = [r for r in records if r["archetype"] == name]
        if len(pool) < per_archetype:
            raise ValueError(f"Only {len(pool)} recorded situations for {name}; need {per_archetype}.")
        picks = rng.choice(len(pool), size=per_archetype, replace=False)
        situations.extend(pool[i] for i in picks)
    for index, situation in enumerate(situations):
        situation["situation"] = index
    return situations


def query_llm(
    situations: List[Dict[str, Any]],
    repeats: int,
    model: str,
    temperature: float,
    base_seed: int,
    completion_fn: Optional[Callable[..., Any]] = None,
) -> pd.DataFrame:
    """Ask the LLM each situation ``repeats`` times (a different seed per repeat).

    ``completion_fn`` replaces the real model call; it exists for testing.
    Returns one row per query. A vote counts as valid only if the output parsed
    (not the hold-after-failure fallback) and names a real faction.
    """
    rows = []
    for situation in situations:
        state = situation["state"]
        candidates = state["candidates"]
        archetype = ARCHETYPES[situation["archetype"]]
        for repeat in range(repeats):
            agent = LLMAgent(
                node_id=state["node_id"],
                candidates=candidates,
                archetype=archetype,
                model=model,
                temperature=temperature,
                seed=base_seed + repeat,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                completion_fn=completion_fn,
            )
            action = agent.decide(state, situation["inbox"])
            legal = action.vote in candidates and (action.target is None or action.target in candidates)
            rows.append(
                {
                    "situation": situation["situation"],
                    "archetype": situation["archetype"],
                    "round": situation["round"],
                    "repeat": repeat,
                    "vote": action.vote,
                    "valid": legal and not agent.last_parse_failed,
                }
            )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Consistency per archetype and overall.

    ``consistency`` is the mean over situations of the share of valid votes that match
    the situation's most common vote (1.0 = every repeat agreed). ``unanimous`` is the
    share of situations where all valid repeats agreed. Situations with fewer than two
    valid votes cannot be judged and are left out (``n_skipped``).
    """
    valid = raw[raw["valid"]]
    per_situation = (
        valid.groupby(["archetype", "situation"])["vote"]
        .agg(n_valid="size", top_count=lambda votes: votes.value_counts().iloc[0])
        .reset_index()
    )
    per_situation = per_situation[per_situation["n_valid"] >= 2]
    per_situation["share"] = per_situation["top_count"] / per_situation["n_valid"]
    per_situation["unanimous"] = per_situation["top_count"] == per_situation["n_valid"]

    def row(label: str, situations: pd.DataFrame, queries: pd.DataFrame) -> Dict[str, Any]:
        return {
            "archetype": label,
            "n_situations": len(situations),
            "n_skipped": queries["situation"].nunique() - len(situations),
            "n_queries": len(queries),
            "invalid_rate": 1 - queries["valid"].mean(),
            "consistency": situations["share"].mean(),
            "unanimous": situations["unanimous"].mean(),
        }

    rows = [
        row(name, per_situation[per_situation["archetype"] == name], raw[raw["archetype"] == name])
        for name in ARCHETYPES
    ]
    rows.append(row("overall", per_situation, raw))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure LLM decision consistency.")
    parser.add_argument("--situations-per-archetype", type=int, default=14)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    config = load_experiment_config()
    base_seed = int(config.get("seed", 42))
    rng = np.random.default_rng(base_seed)

    print("Collecting situations from baseline trials...")
    records = asyncio.run(
        collect_states(
            num_agents=int(config.get("agents", 5)),
            max_rounds=int(config.get("max_rounds", 20)),
            fanout=int(config.get("fanout", 1)),
            round_ms=float(config.get("round_ms", 1000.0)),
            base_seed=base_seed,
        )
    )
    situations = sample_situations(records, args.situations_per_archetype, rng)
    total = len(situations) * args.repeats
    print(f"Querying the LLM: {len(situations)} situations x {args.repeats} repeats = {total} calls...")
    raw = query_llm(
        situations,
        args.repeats,
        model=str(config.get("model", DEFAULT_MODEL)),
        temperature=float(config.get("temperature", DEFAULT_TEMPERATURE)),
        base_seed=base_seed,
    )

    summary = summarize(raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "decision_consistency_raw.csv", index=False)
    summary.to_csv(out_dir / "decision_consistency.csv", index=False)
    print("\n=== decision_consistency ===")
    print(summary.to_string(index=False, float_format="{:.4g}".format))
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()