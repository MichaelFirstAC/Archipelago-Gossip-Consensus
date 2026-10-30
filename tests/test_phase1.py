"""Automated verification suite for Phase 1: Network Layer with Dummy Agents."""

import asyncio
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network.topology import build_hub_spoke, build_mesh
from src.network.node import Node
from src.simulation.consensus import check_consensus, is_converged
from src.simulation.runner import run_trial

# Cooperative-only mix and a generous round cap make zero-loss convergence deterministic for the seeds used.
COOP = {"cooperative": 1.0}
SEEDS = range(5)


def test_topologies():
    print("[TEST] Verifying topologies...")
    mesh = build_mesh(5)
    assert len(mesh) == 5, f"Expected 5 nodes, got {len(mesh)}"
    for nid, neighbors in mesh.items():
        assert len(neighbors) == 4, f"Mesh node {nid} should have 4 neighbors, got {len(neighbors)}"
        assert nid not in neighbors, f"Node {nid} has self-loop"

    hub_spoke = build_hub_spoke(5)
    assert len(hub_spoke) == 5
    assert len(hub_spoke["faction_0"]) == 4, "Hub should connect to 4 spokes"
    for i in range(1, 5):
        spoke_id = f"faction_{i}"
        assert hub_spoke[spoke_id] == ["faction_0"], f"Spoke {spoke_id} should only connect to hub"
    print("  -> Topologies verified successfully.")


def test_consensus_logic():
    print("[TEST] Verifying consensus logic...")
    same = np.array([0.5, 0.3, 0.2])
    nodes = [Node(f"n{i}", initial_belief="faction_0", belief_vector=same.copy()) for i in range(3)]
    assert is_converged(nodes) is True
    conv, val = check_consensus(nodes)
    assert conv is True and val == "faction_0"

    nodes[2].vote = "faction_1"  # vote disagreement blocks consensus
    assert is_converged(nodes) is False
    conv, val = check_consensus(nodes)
    assert conv is False and val is None

    nodes[2].vote = "faction_0"
    nodes[2].belief_vector = np.array([0.1, 0.1, 0.8])  # same vote but beliefs far apart
    assert is_converged(nodes) is False
    print("  -> Consensus logic verified successfully.")


async def test_trials_and_logging():
    print("[TEST] Verifying trial execution, convergence, and logging...")
    test_log_dir = "data/runs_test"
    if Path(test_log_dir).exists():
        shutil.rmtree(test_log_dir)

    print("  Testing 5 trials at latency=0, loss=0 on mesh...")
    for i in SEEDS:
        res = await run_trial(
            trial_id=f"test_mesh_zero_loss_{i}", latency_ms=0.0, loss_rate=0.0, topology="mesh",
            num_agents=5, max_rounds=60, archetype_mix=COOP, seed=i, log_dir=test_log_dir,
        )
        assert res["converged"] is True, f"Mesh trial {i} failed to reach consensus!"
        assert 0 < res["rounds_to_consensus"] <= 60
        assert res["parse_failures"] == 0

    print("  Testing 5 trials at latency=0, loss=0 on hub_spoke...")
    for i in SEEDS:
        res = await run_trial(
            trial_id=f"test_hub_zero_loss_{i}", latency_ms=0.0, loss_rate=0.0, topology="hub_spoke",
            num_agents=5, max_rounds=60, archetype_mix=COOP, seed=i, log_dir=test_log_dir,
        )
        assert res["converged"] is True, f"Hub & Spoke trial {i} failed to reach consensus!"

    print("  Testing 5 trials at latency=0, loss=1.0 (drop all)...")
    for i in SEEDS:
        res = await run_trial(
            trial_id=f"test_total_loss_{i}", latency_ms=0.0, loss_rate=1.0, topology="mesh",
            num_agents=5, max_rounds=15, seed=i, log_dir=test_log_dir,
        )
        assert res["converged"] is False, f"Trial {i} unexpectedly converged despite 100% loss!"
        assert res["rounds_to_consensus"] == 15

    sample_event_csv = Path(test_log_dir) / "events_test_mesh_zero_loss_0.csv"
    assert sample_event_csv.exists(), f"Event CSV {sample_event_csv} does not exist"
    with open(sample_event_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) > 0, "Event CSV is empty"
        for column in ("trial_id", "round", "sender", "receiver", "latency_ms", "dropped", "delivered_at"):
            assert column in reader.fieldnames
        for r in rows:
            assert r["dropped"] == "False"
            assert r["delivered_at"] != ""

    drop_event_csv = Path(test_log_dir) / "events_test_total_loss_0.csv"
    assert drop_event_csv.exists()
    with open(drop_event_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) > 0
        for r in rows:
            assert r["dropped"] == "True"
            assert r["delivered_at"] == ""

    summary_csv = Path(test_log_dir) / "trials_summary.csv"
    assert summary_csv.exists(), "Master summary CSV does not exist"
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        summary_rows = list(reader)
        assert len(summary_rows) == 15, f"Expected 15 summary rows, found {len(summary_rows)}"
        for column in ("converged", "rounds_to_consensus", "total_messages"):
            assert column in reader.fieldnames

    shutil.rmtree(test_log_dir)
    print("  -> Trial execution, convergence criteria, and CSV logging fully verified!")


def main():
    test_topologies()
    test_consensus_logic()
    asyncio.run(test_trials_and_logging())
    print("\nALL PHASE 1 SANITY CHECKS AND VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
