"""Monte Carlo batch runner: sweeps config grid, runs N trials per condition, writes logs."""

import argparse
import asyncio
import csv
import itertools
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from src.agents.archetypes import DEFAULT_MIX
from src.agents.llm_agent import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from src.logging.run_logger import SUMMARY_FIELDNAMES
from src.simulation.consensus import DEFAULT_EPSILON
from src.simulation.runner import run_trial

PathLike = Union[str, Path]


def load_experiment_config(config_path: PathLike = "config/experiment.yaml") -> Dict[str, Any]:
    """Load and parse the YAML experiment configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment configuration file not found: {path}")
    with open(path, mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def expand_grid_section(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a single grid section (e.g. primary_grid or secondary_grid) into condition dicts."""
    if not section:
        return []
    latencies = section.get("latency_ms", [0])
    loss_rates = section.get("loss_rate", [0.0])
    topologies = section.get("topology", ["mesh"])

    conditions: List[Dict[str, Any]] = []
    for topo, lat, loss in itertools.product(topologies, latencies, loss_rates):
        conditions.append(
            {
                "topology": str(topo),
                "latency_ms": float(lat),
                "loss_rate": float(loss),
            }
        )
    return conditions


def get_conditions(config: Dict[str, Any], grid_mode: str = "all") -> List[Dict[str, Any]]:
    """Return deduplicated list of condition dicts for the requested grid mode.

    Args:
        config: Parsed experiment YAML dictionary.
        grid_mode: 'primary' (9 mesh cells), 'secondary' (only non-overlapping secondary cells,
            or all secondary cells if primary is empty), or 'all' (primary + deduplicated secondary).

    Returns:
        Ordered list of unique condition dictionaries with keys ('topology', 'latency_ms', 'loss_rate').
    """
    primary = expand_grid_section(config.get("primary_grid", {}))
    secondary = expand_grid_section(config.get("secondary_grid", {}))

    if grid_mode == "primary":
        raw_list = primary
    elif grid_mode == "secondary":
        primary_keys = {(c["topology"], c["latency_ms"], round(c["loss_rate"], 6)) for c in primary}
        non_overlapping = [
            c for c in secondary if (c["topology"], c["latency_ms"], round(c["loss_rate"], 6)) not in primary_keys
        ]
        raw_list = non_overlapping if non_overlapping else secondary
    elif grid_mode == "all":
        raw_list = primary + secondary
    else:
        raise ValueError(f"Unknown grid_mode: {grid_mode!r}. Expected 'primary', 'secondary', or 'all'.")

    seen: Set[Tuple[str, float, float]] = set()
    deduped: List[Dict[str, Any]] = []
    for cond in raw_list:
        key = (cond["topology"], float(cond["latency_ms"]), round(float(cond["loss_rate"]), 6))
        if key not in seen:
            seen.add(key)
            deduped.append(cond)
    return deduped


def format_trial_id(
    agent_type: str,
    topology: str,
    latency_ms: float,
    loss_rate: float,
    run_idx: int,
) -> str:
    """Build a deterministic, human-readable trial_id for a grid cell and run index."""
    lat_int = int(round(latency_ms))
    loss_pct = int(round(loss_rate * 100))
    return f"{agent_type}_{topology}_lat{lat_int}_loss{loss_pct:02d}_run{run_idx:03d}"


def get_completed_trials(
    log_dir: PathLike,
    master_summary_filename: str = "trials_summary.csv",
) -> Set[str]:
    """Inspect `log_dir` and return the set of fully completed trial_ids.

    Also deduplicates `trials_summary.csv` in-place if any interrupted run left duplicate
    or incomplete summary rows (where `events_<trial_id>.csv` or `decisions_<trial_id>.csv`
    is missing or empty).
    """
    out_dir = Path(log_dir)
    summary_path = out_dir / master_summary_filename
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return set()

    valid_rows_by_id: Dict[str, Dict[str, Any]] = {}
    needs_rewrite = False
    total_rows_read = 0

    with open(summary_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows_read += 1
            tid = row.get("trial_id", "").strip()
            if not tid:
                needs_rewrite = True
                continue
            events_file = out_dir / f"events_{tid}.csv"
            decisions_file = out_dir / f"decisions_{tid}.csv"
            has_files = (
                events_file.exists()
                and events_file.stat().st_size > 0
                and decisions_file.exists()
                and decisions_file.stat().st_size > 0
            )
            if not has_files:
                needs_rewrite = True
                continue
            if tid in valid_rows_by_id:
                needs_rewrite = True
            valid_rows_by_id[tid] = row

    if needs_rewrite or len(valid_rows_by_id) != total_rows_read:
        with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(valid_rows_by_id.values())

    return set(valid_rows_by_id.keys())


async def run_sweep(
    config_path: PathLike = "config/experiment.yaml",
    agent_types: Optional[List[str]] = None,
    grid_mode: Optional[str] = None,
    runs_override: Optional[int] = None,
    log_dir: PathLike = "data/runs",
    master_summary_filename: str = "trials_summary.csv",
    time_scale: float = 0.02,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a resumable Monte Carlo sweep across the configured conditions.

    Args:
        config_path: Path to YAML config file.
        agent_types: List of agent types to run (e.g. ['baseline'], ['llm'], or ['baseline', 'llm']).
            Defaults to `[config.get('agent_type', 'llm')]`.
        grid_mode: 'primary', 'secondary', or 'all'. If None, defaults to 'primary' for
            baseline and 'all' (primary + secondary topology subset) for llm/dummy.
        runs_override: Optional integer overriding `runs_per_condition` / `baseline_runs_per_condition`.
        log_dir: Output directory for CSV logs.
        master_summary_filename: Summary CSV filename inside `log_dir`.
        verbose: Whether to print progress updates.

    Returns:
        Dictionary with sweep statistics (`executed`, `skipped`, `total_planned`, `summaries`).
    """
    config = load_experiment_config(config_path)

    num_agents = int(config.get("agents", 5))
    max_rounds = int(config.get("max_rounds", 20))
    fanout = int(config.get("fanout", 1))
    round_ms = float(config.get("round_ms", 1000.0))
    model = str(config.get("model", DEFAULT_MODEL))
    temperature = float(config.get("temperature", DEFAULT_TEMPERATURE))
    base_seed = int(config.get("seed", 42))
    epsilon = float(config.get("epsilon", DEFAULT_EPSILON))
    archetype_mix = config.get("archetype_mix") or DEFAULT_MIX

    if not agent_types:
        agent_types = [str(config.get("agent_type", "llm"))]

    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    completed_ids = get_completed_trials(out_dir, master_summary_filename=master_summary_filename)

    # Build full work plan first so we can report accurate progress and ETA
    plan: List[Tuple[str, Dict[str, Any], int, str]] = []
    for ag_type in agent_types:
        effective_grid = grid_mode if grid_mode is not None else ("primary" if ag_type == "baseline" else "all")
        conditions = get_conditions(config, grid_mode=effective_grid)

        if runs_override is not None:
            n_runs = int(runs_override)
        elif ag_type == "baseline":
            n_runs = int(config.get("baseline_runs_per_condition", config.get("runs_per_condition", 30)))
        else:
            n_runs = int(config.get("runs_per_condition", 15))

        for cond in conditions:
            for run_idx in range(n_runs):
                tid = format_trial_id(
                    agent_type=ag_type,
                    topology=cond["topology"],
                    latency_ms=cond["latency_ms"],
                    loss_rate=cond["loss_rate"],
                    run_idx=run_idx,
                )
                plan.append((ag_type, cond, run_idx, tid))

    total_planned = len(plan)
    executed = 0
    skipped = 0
    summaries: List[Dict[str, Any]] = []
    sweep_start = time.perf_counter()

    if verbose:
        pending_count = sum(1 for _, _, _, tid in plan if tid not in completed_ids)
        print(
            f"[MonteCarlo] Planned: {total_planned} trials | "
            f"Already completed: {total_planned - pending_count} | "
            f"To run: {pending_count}"
        )

    for idx, (ag_type, cond, run_idx, tid) in enumerate(plan, start=1):
        if tid in completed_ids:
            skipped += 1
            continue

        trial_seed = base_seed + run_idx
        t0 = time.perf_counter()
        summary = await run_trial(
            trial_id=tid,
            latency_ms=cond["latency_ms"],
            loss_rate=cond["loss_rate"],
            topology=cond["topology"],
            num_agents=num_agents,
            max_rounds=max_rounds,
            fanout=fanout,
            round_ms=round_ms,
            archetype_mix=archetype_mix,
            agent_type=ag_type,
            model=model,
            temperature=temperature,
            epsilon=epsilon,
            seed=trial_seed,
            log_dir=str(out_dir),
            save_logs=True,
            master_summary_filename=master_summary_filename,
            time_scale=time_scale,
        )
        dt = time.perf_counter() - t0
        executed += 1
        completed_ids.add(tid)
        summaries.append(summary)

        if verbose:
            elapsed = time.perf_counter() - sweep_start
            avg_per_exec = elapsed / executed
            remaining_to_run = sum(1 for _, _, _, rem_tid in plan[idx:] if rem_tid not in completed_ids)
            eta_s = avg_per_exec * remaining_to_run
            print(
                f"[{idx}/{total_planned}] {tid} -> "
                f"converged={summary['converged']} rounds={summary['rounds_to_consensus']} "
                f"msgs={summary['total_messages']} fail={summary['parse_failures']} "
                f"({dt:.1f}s, ETA {eta_s/60:.1f}m)",
                flush=True,
            )

    if verbose:
        total_elapsed = time.perf_counter() - sweep_start
        print(
            f"[MonteCarlo] Done in {total_elapsed:.1f}s | "
            f"Executed: {executed} | Skipped (resumed): {skipped} | Total: {total_planned}"
        )

    return {
        "total_planned": total_planned,
        "executed": executed,
        "skipped": skipped,
        "summaries": summaries,
    }


def main() -> None:
    """CLI entrypoint for the Monte Carlo batch experiment runner."""
    parser = argparse.ArgumentParser(description="Run Archipelago Monte Carlo batch experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment.yaml",
        help="Path to experiment YAML config (default: config/experiment.yaml).",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default=None,
        choices=["baseline", "llm", "dummy", "both"],
        help="Agent arm to run ('baseline', 'llm', 'dummy', or 'both' for baseline+llm).",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        choices=["primary", "secondary", "all"],
        help="Grid subset to expand ('primary', 'secondary', or 'all').",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override number of runs per condition (e.g. --runs 2 for pilot sweep).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="data/runs",
        help="Directory to write trial summaries, events, and decisions.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-trial progress output.",
    )

    args = parser.parse_args()

    if args.agent_type == "both":
        agent_types = ["baseline", "llm"]
    elif args.agent_type is not None:
        agent_types = [args.agent_type]
    else:
        agent_types = None

    asyncio.run(
        run_sweep(
            config_path=args.config,
            agent_types=agent_types,
            grid_mode=args.grid,
            runs_override=args.runs,
            log_dir=args.log_dir,
            verbose=not args.quiet,
        )
    )


if __name__ == "__main__":
    main()
