"""Automated verification suite for Phase 2A: baseline agent and belief module."""

import asyncio
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import belief as bel
from src.agents.archetypes import ARCHETYPES, DEFAULT_MIX, assign_archetypes
from src.agents.baseline_agent import BaselineAgent
from src.agents.factory import create_agent
from src.agents.dummy_agent import DummyAgent
from src.logging.run_logger import DECISION_FIELDNAMES, SUMMARY_FIELDNAMES
from src.simulation.runner import run_trial

COOP = {"cooperative": 1.0}
LOG_DIR = "data/runs_test_2a"


def test_degroot_hand_example():
    print("[TEST] DeGroot update matches the hand calculation...")
    # Node 0 (belief [1,0,0], openness 0.5) hears node 1 ([0,1,0]) and node 2 ([0,0,1]).
    # Trust 1 and 3 normalizes to T = [0.25, 0.75]; neighbor mix = [0, 0.25, 0.75].
    # New belief = 0.5*[1,0,0] + 0.5*[0, 0.25, 0.75] = [0.5, 0.125, 0.375].
    own = np.array([1.0, 0.0, 0.0])
    received = [("n1", np.array([0.0, 1.0, 0.0])), ("n2", np.array([0.0, 0.0, 1.0]))]
    updated = bel.degroot_update(own, received, openness=0.5, trust={"n1": 1.0, "n2": 3.0})
    assert np.allclose(updated, [0.5, 0.125, 0.375]), updated
    assert abs(updated.sum() - 1.0) < 1e-12

    # No messages (all dropped): belief is unchanged.
    assert np.allclose(bel.degroot_update(own, [], openness=0.5), own)
    # Fully stubborn (openness 0) ignores neighbors; fully open (1) copies the neighbor mix.
    assert np.allclose(bel.degroot_update(own, received, openness=0.0), own)
    assert np.allclose(bel.degroot_update(own, received, openness=1.0), [0.0, 0.5, 0.5])
    print("  -> OK")


def test_consensus_score():
    print("[TEST] Consensus score is max pairwise total-variation distance...")
    a, b, c = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0.5, 0.5, 0])
    assert bel.total_variation(a, a) == 0.0
    assert bel.total_variation(a, b) == 1.0
    assert abs(bel.total_variation(a, c) - 0.5) < 1e-12
    assert bel.consensus_score([a, a, a]) == 0.0
    assert bel.consensus_score([a, b, c]) == 1.0
    assert bel.consensus_score([a]) == 0.0
    print("  -> OK")


def test_initial_belief():
    print("[TEST] Initial beliefs are valid distributions leaning toward own faction...")
    rng = np.random.default_rng(0)
    for i in range(5):
        v = bel.initial_belief(5, i, rng)
        assert abs(v.sum() - 1.0) < 1e-9 and (v >= 0).all()
    print("  -> OK")


def test_archetype_assignment():
    print("[TEST] Archetype assignment follows the mix...")
    ids = [f"faction_{i}" for i in range(5)]
    for seed in range(5):
        assigned = assign_archetypes(ids, DEFAULT_MIX, np.random.default_rng(seed))
        names = sorted(a.name for a in assigned.values())
        assert names == ["adversarial", "cooperative", "cooperative", "cooperative", "self_interested"], names
    only_coop = assign_archetypes(ids, COOP, np.random.default_rng(0))
    assert {a.name for a in only_coop.values()} == {"cooperative"}
    print("  -> OK")


def test_baseline_agent():
    print("[TEST] Baseline agent is deterministic and returns a valid Action...")
    candidates = ["faction_0", "faction_1", "faction_2"]
    state = {
        "node_id": "faction_0", "vote": "faction_0", "belief": "faction_0",
        "belief_vector": [0.2, 0.7, 0.1], "neighbors": ["faction_1"], "candidates": candidates,
    }
    inbox = [{"sender": "faction_1", "payload": {"belief_vector": [0.1, 0.8, 0.1]}}]
    agent = BaselineAgent("faction_0", candidates, ARCHETYPES["cooperative"])
    first, second = agent.decide(state, inbox), agent.decide(state, inbox)
    assert first == second
    assert first.vote == "faction_1" and first.action == "forge_alliance" and first.target == "faction_1"

    # A stubborn, self-serving faction with a flat belief keeps voting for itself.
    flat = {**state, "belief_vector": [0.34, 0.33, 0.33]}
    adversary = BaselineAgent("faction_0", candidates, ARCHETYPES["adversarial"])
    assert adversary.decide(flat, []).vote == "faction_0"
    assert adversary.decide(flat, []).action == "hold"
    print("  -> OK")


def test_factory():
    print("[TEST] Factory builds the requested agent type...")
    c = ["faction_0", "faction_1"]
    assert isinstance(create_agent("dummy", "faction_0", c, ARCHETYPES["cooperative"]), DummyAgent)
    assert isinstance(create_agent("baseline", "faction_0", c, ARCHETYPES["cooperative"]), BaselineAgent)
    for bad in ("nope",):
        try:
            create_agent(bad, "faction_0", c, ARCHETYPES["cooperative"])
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    print("  -> OK")


async def test_baseline_trials():
    print("[TEST] Baseline trials: zero loss converges, total loss never does...")
    if Path(LOG_DIR).exists():
        shutil.rmtree(LOG_DIR, ignore_errors=True)

    for seed in range(10):
        res = await run_trial(f"coop_{seed}", loss_rate=0.0, topology="mesh", agent_type="baseline",
                              max_rounds=30, archetype_mix=COOP, seed=seed, log_dir=LOG_DIR)
        assert res["converged"], f"cooperative baseline trial {seed} did not converge in 30 rounds"

    converged = 0
    for seed in range(100):
        res = await run_trial(f"mix_{seed}", loss_rate=0.0, topology="mesh", agent_type="baseline",
                              seed=seed, save_logs=False)
        converged += res["converged"]
    print(f"  default mix, mesh, loss=0, 20 rounds: {converged}/100 converged")
    assert converged >= 90, f"only {converged}/100 converged at zero loss"

    for seed in range(10):
        res = await run_trial(f"drop_{seed}", loss_rate=1.0, topology="mesh", agent_type="baseline",
                              seed=seed, log_dir=LOG_DIR)
        assert not res["converged"], f"baseline trial {seed} converged despite 100% loss"

    # same seed -> same result
    a = await run_trial("rep_a", loss_rate=0.3, agent_type="baseline", seed=7, save_logs=False)
    b = await run_trial("rep_b", loss_rate=0.3, agent_type="baseline", seed=7, save_logs=False)
    assert (a["converged"], a["rounds_to_consensus"]) == (b["converged"], b["rounds_to_consensus"])
    print("  -> OK")


async def test_logs():
    print("[TEST] Decision log and trial summary have the new columns...")
    decisions = Path(LOG_DIR) / "decisions_coop_0.csv"
    with open(decisions, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == DECISION_FIELDNAMES
    assert rows and all(r["agent_type"] == "baseline" for r in rows)
    assert all(r["baseline_vote"] == r["vote"] for r in rows), "baseline arm must agree with itself"
    assert all(r["valid_json"] == "True" and r["valid_action"] == "True" for r in rows)

    with open(Path(LOG_DIR) / "trials_summary.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        summary = list(reader)
    assert reader.fieldnames == SUMMARY_FIELDNAMES
    assert len(summary) == 20 and all(r["agent_type"] == "baseline" for r in summary)

    # baseline_vote is filled in for a non-baseline arm too (used later for LLM-vs-baseline agreement)
    await run_trial("dummy_log", loss_rate=0.0, agent_type="dummy", max_rounds=5, seed=1, log_dir=LOG_DIR)
    with open(Path(LOG_DIR) / "decisions_dummy_log.csv", encoding="utf-8") as f:
        dummy_rows = list(csv.DictReader(f))
    assert dummy_rows and all(r["baseline_vote"] != "" and r["agent_type"] == "dummy" for r in dummy_rows)

    shutil.rmtree(LOG_DIR, ignore_errors=True)
    print("  -> OK")


def main():
    test_degroot_hand_example()
    test_consensus_score()
    test_initial_belief()
    test_archetype_assignment()
    test_baseline_agent()
    test_factory()
    asyncio.run(_trials_and_logs())
    print("\nALL PHASE 2A TESTS PASSED SUCCESSFULLY!")


async def _trials_and_logs():
    await test_baseline_trials()
    await test_logs()


if __name__ == "__main__":
    main()
