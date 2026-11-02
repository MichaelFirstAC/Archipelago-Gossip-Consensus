"""Automated verification suite for Phase 3: Batch Experiment Runner (monte_carlo.py)."""

import asyncio
import csv
import shutil
import sys
from pathlib import Path

import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging.run_logger import SUMMARY_FIELDNAMES
from src.simulation.monte_carlo import (
    format_trial_id,
    get_completed_trials,
    get_conditions,
    load_experiment_config,
    run_sweep,
)

TEST_DIR = Path("data/runs_test_phase3")


def test_grid_expansion_and_deduplication():
    print("[TEST] Grid expansion and primary/secondary deduplication...")
    cfg = load_experiment_config("config/experiment.yaml")

    primary = get_conditions(cfg, grid_mode="primary")
    assert len(primary) == 9, f"Expected 9 primary conditions, got {len(primary)}"
    assert all(c["topology"] == "mesh" for c in primary)

    secondary_only = get_conditions(cfg, grid_mode="secondary")
    assert len(secondary_only) == 4, f"Expected 4 non-overlapping secondary conditions, got {len(secondary_only)}"
    assert all(c["topology"] == "hub_spoke" for c in secondary_only)

    all_conds = get_conditions(cfg, grid_mode="all")
    assert len(all_conds) == 13, f"Expected 13 deduplicated total conditions, got {len(all_conds)}"

    tid = format_trial_id("llm", "hub_spoke", 2000.0, 0.15, 7)
    assert tid == "llm_hub_spoke_lat2000_loss15_run007", tid
    print("  -> OK")


async def test_resumability_and_cleanup():
    print("[TEST] Resumability: interrupted sweeps do not overwrite or repeat finished trials...")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Create a tiny fast config (0ms and 10ms latency) so unit tests run in <1s
    mini_cfg = {
        "agents": 5,
        "max_rounds": 5,
        "fanout": 1,
        "round_ms": 50,
        "runs_per_condition": 2,
        "baseline_runs_per_condition": 2,
        "agent_type": "baseline",
        "model": "ollama/llama3.2:3b",
        "temperature": 0.7,
        "seed": 42,
        "epsilon": 0.05,
        "primary_grid": {
            "latency_ms": [0, 10],
            "loss_rate": [0.0, 0.30],
            "topology": ["mesh"],
        },
        "secondary_grid": {
            "latency_ms": [0],
            "loss_rate": [0.0],
            "topology": ["mesh", "hub_spoke"],
        },
    }
    cfg_path = TEST_DIR / "mini_experiment.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mini_cfg, f)

    # 1. Initial sweep: baseline on primary grid (4 conditions * 2 runs = 8 trials)
    res1 = await run_sweep(
        config_path=cfg_path,
        agent_types=["baseline"],
        grid_mode="primary",
        runs_override=2,
        log_dir=TEST_DIR,
        verbose=False,
    )
    assert res1["total_planned"] == 8
    assert res1["executed"] == 8
    assert res1["skipped"] == 0

    # 2. Re-run exact same sweep: all 8 trials must be skipped
    res2 = await run_sweep(
        config_path=cfg_path,
        agent_types=["baseline"],
        grid_mode="primary",
        runs_override=2,
        log_dir=TEST_DIR,
        verbose=False,
    )
    assert res2["total_planned"] == 8
    assert res2["executed"] == 0
    assert res2["skipped"] == 8

    # 3. Simulate an interrupted/corrupted trial: delete one decision file and add a duplicate summary row
    corrupted_tid = "baseline_mesh_lat0_loss00_run001"
    (TEST_DIR / f"decisions_{corrupted_tid}.csv").unlink()

    summary_csv = TEST_DIR / "trials_summary.csv"
    with open(summary_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Append a duplicate of row 0
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writerow(rows[0])

    completed = get_completed_trials(TEST_DIR)
    assert len(completed) == 7, f"Expected 7 valid completed trials after corruption, got {len(completed)}"
    assert corrupted_tid not in completed

    # 4. Expand sweep to 3 runs per condition (4 * 3 = 12 planned):
    #    7 already completed -> 1 repaired + 4 new = 5 executed
    res3 = await run_sweep(
        config_path=cfg_path,
        agent_types=["baseline"],
        grid_mode="primary",
        runs_override=3,
        log_dir=TEST_DIR,
        verbose=False,
    )
    assert res3["total_planned"] == 12
    assert res3["skipped"] == 7
    assert res3["executed"] == 5

    with open(summary_csv, "r", encoding="utf-8") as f:
        final_rows = list(csv.DictReader(f))
    assert len(final_rows) == 12, f"Expected 12 unique summary rows, got {len(final_rows)}"
    assert len({r["trial_id"] for r in final_rows}) == 12

    shutil.rmtree(TEST_DIR, ignore_errors=True)
    print("  -> OK")


def main():
    test_grid_expansion_and_deduplication()
    asyncio.run(test_resumability_and_cleanup())
    print("\nALL PHASE 3 UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

