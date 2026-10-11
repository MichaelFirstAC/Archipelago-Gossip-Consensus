"""Automated verification suite for Phase 1: Network Layer with Dummy Agents."""

import asyncio
import csv
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network.topology import build_hub_spoke, build_mesh
from src.network.node import Node
from src.network.link import Link
from src.simulation.consensus import check_consensus, is_converged
from src.simulation.runner import run_trial


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
    n1 = Node("n1", initial_belief="faction_0")
    n2 = Node("n2", initial_belief="faction_0")
    n3 = Node("n3", initial_belief="faction_0")
    assert is_converged([n1, n2, n3]) is True
    conv, val = check_consensus([n1, n2, n3])
    assert conv is True and val == "faction_0"

    n3.belief = "faction_1"
    assert is_converged([n1, n2, n3]) is False
    conv, val = check_consensus([n1, n2, n3])
    assert conv is False and val is None
    print("  -> Consensus logic verified successfully.")


async def test_trials_and_logging():
    print("[TEST] Verifying trial execution, convergence, and logging...")
    test_log_dir = "data/runs_test"
    if Path(test_log_dir).exists():
        shutil.rmtree(test_log_dir)

    # 1. Test loss=0.0 on Mesh (Must converge every time)
    print("  Testing 5 trials at latency=0, loss=0 on mesh...")
    for i in range(5):
        trial_id = f"test_mesh_zero_loss_{i}"
        res = await run_trial(
            trial_id=trial_id,
            latency_ms=0.0,
            loss_rate=0.0,
            topology="mesh",
            num_agents=5,
            max_rounds=15,
            log_dir=test_log_dir,
            save_logs=True,
        )
        assert res["converged"] is True, f"Trial {trial_id} failed to reach consensus!"
        assert res["rounds_to_consensus"] > 0
        assert res["rounds_to_consensus"] <= 15
        assert res["parse_failures"] == 0

    # 2. Test loss=0.0 on Hub & Spoke (Must converge)
    print("  Testing 5 trials at latency=0, loss=0 on hub_spoke...")
    for i in range(5):
        trial_id = f"test_hub_zero_loss_{i}"
        res = await run_trial(
            trial_id=trial_id,
            latency_ms=0.0,
            loss_rate=0.0,
            topology="hub_spoke",
            num_agents=5,
            max_rounds=15,
            log_dir=test_log_dir,
            save_logs=True,
        )
        assert res["converged"] is True, f"Hub & Spoke trial {trial_id} failed to reach consensus!"

    # 3. Test loss=1.0 (Must NEVER converge)
    print("  Testing 5 trials at latency=0, loss=1.0 (drop all)...")
    for i in range(5):
        trial_id = f"test_total_loss_{i}"
        res = await run_trial(
            trial_id=trial_id,
            latency_ms=0.0,
            loss_rate=1.0,
            topology="mesh",
            num_agents=5,
            max_rounds=15,
            log_dir=test_log_dir,
            save_logs=True,
        )
        assert res["converged"] is False, f"Trial {trial_id} unexpectedly converged despite 100% loss!"
        assert res["rounds_to_consensus"] == 15

    # 4. Validate Event Log CSV
    sample_event_csv = Path(test_log_dir) / "events_test_mesh_zero_loss_0.csv"
    assert sample_event_csv.exists(), f"Event CSV {sample_event_csv} does not exist"
    with open(sample_event_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) > 0, "Event CSV is empty"
        assert "trial_id" in reader.fieldnames
        assert "round" in reader.fieldnames
        assert "sender" in reader.fieldnames
        assert "receiver" in reader.fieldnames
        assert "latency_ms" in reader.fieldnames
        assert "dropped" in reader.fieldnames
        assert "delivered_at" in reader.fieldnames
        # Under loss=0, all should be dropped=False
        for r in rows:
            assert r["dropped"] == "False"
            assert r["delivered_at"] != ""

    # Validate 100% drop event CSV
    drop_event_csv = Path(test_log_dir) / "events_test_total_loss_0.csv"
    assert drop_event_csv.exists()
    with open(drop_event_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) > 0
        for r in rows:
            assert r["dropped"] == "True"
            assert r["delivered_at"] == ""

    # 5. Validate Master Summary CSV
    summary_csv = Path(test_log_dir) / "trials_summary.csv"
    assert summary_csv.exists(), "Master summary CSV does not exist"
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        summary_rows = list(reader)
        # Total trials run: 5 + 5 + 5 = 15
        assert len(summary_rows) == 15, f"Expected 15 summary rows, found {len(summary_rows)}"
        assert "converged" in reader.fieldnames
        assert "rounds_to_consensus" in reader.fieldnames
        assert "total_messages" in reader.fieldnames

    # Clean up test output dir
    shutil.rmtree(test_log_dir)
    print("  -> Trial execution, convergence criteria, and CSV logging fully verified!")


def main():
    test_topologies()
    test_consensus_logic()
    asyncio.run(test_trials_and_logging())
    print("\nALL PHASE 1 SANITY CHECKS AND VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
