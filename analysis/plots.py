"""Generates the figures for the Results section.

Run from the repo root:
    python -m analysis.plots [--runs-dir data/runs] [--out-dir results/figures]

Figures (PNG):
    failure_heatmap       failure rate by latency x loss (LLM and baseline, mesh)
    rounds_vs_loss        rounds to consensus vs loss, one panel per latency (LLM, mesh)
    topology_comparison   mesh vs hub_spoke failure rate and rounds (LLM, matched conditions)
    llm_vs_baseline       convergence rate and rounds per condition (mesh)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.load_logs import DEFAULT_RUNS_DIR, llm_mesh, load_trials, topology_matched

DEFAULT_OUT_DIR = "results/figures"
MAX_ROUNDS = 20
SEED = 0  # fixes the bootstrap confidence intervals so figures are reproducible
AGENT_LABELS = {"llm": "LLM agent", "baseline": "Baseline agent"}
TOPOLOGY_LABELS = {"mesh": "Mesh", "hub_spoke": "Hub-and-spoke"}

sns.set_theme(style="whitegrid", palette="colorblind")


def _add_condition(trials: pd.DataFrame) -> pd.DataFrame:
    """Add a ``condition`` label such as '2000 ms / 30%' for plot axes."""
    trials = trials.copy()
    loss_pct = (trials["loss_rate"] * 100).round().astype(int)
    trials["condition"] = trials["latency"].astype(int).astype(str) + " ms\n" + loss_pct.astype(str) + "%"
    return trials


def _condition_order(trials: pd.DataFrame) -> list:
    """Condition labels ordered by latency, then loss."""
    return trials.sort_values(["latency", "loss_rate"])["condition"].unique().tolist()


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def failure_heatmap(trials: pd.DataFrame, out_dir: Path) -> None:
    """Failure rate (%) by latency x loss, LLM and baseline side by side on one scale."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, agent in zip(axes, ["llm", "baseline"]):
        mesh = trials[(trials["agent_type"] == agent) & (trials["topology"] == "mesh")]
        rate = 100 * mesh.pivot_table(index="latency", columns="loss_rate", values="failed", aggfunc="mean")
        rate.index = [f"{int(latency)} ms" for latency in rate.index]
        rate.columns = [f"{loss:.0%}" for loss in rate.columns]
        runs = mesh.groupby(["latency", "loss_rate"]).size().iloc[0]
        sns.heatmap(
            rate,
            annot=True,
            fmt=".0f",
            cmap="Reds",
            vmin=0,
            vmax=100,
            ax=ax,
            cbar=agent == "baseline",
            cbar_kws={"label": "Failure rate (%)"},
        )
        ax.invert_yaxis()  # latency increases upward
        ax.set_title(f"{AGENT_LABELS[agent]} (n={runs} per cell)")
        ax.set_xlabel("Packet loss")
        ax.set_ylabel("Latency" if agent == "llm" else "")
    fig.suptitle("Consensus failure rate (mesh)", y=1.02)
    _save(fig, out_dir, "failure_heatmap")


def rounds_vs_loss(trials: pd.DataFrame, out_dir: Path) -> None:
    """Mean rounds to consensus vs loss, one panel per latency (LLM, mesh, converged trials)."""
    converged = llm_mesh(trials).dropna(subset=["rounds_converged"]).copy()
    converged["loss_pct"] = (converged["loss_rate"] * 100).round().astype(int).astype(str) + "%"
    grid = sns.catplot(
        data=converged,
        x="loss_pct",
        y="rounds_converged",
        col="latency",
        kind="point",
        order=["0%", "15%", "30%"],
        errorbar=("ci", 95),
        seed=SEED,
        capsize=0.1,
        height=3.6,
        aspect=0.9,
    )
    grid.set_titles("Latency {col_name:.0f} ms")
    grid.set_axis_labels("Packet loss", "Rounds to consensus")
    grid.set(ylim=(0, MAX_ROUNDS))
    grid.figure.suptitle("Time to consensus (LLM, mesh; converged trials only; mean and 95% CI)", y=1.04)
    _save(grid.figure, out_dir, "rounds_vs_loss")


def _shared_legend(axes: list) -> None:
    """Replace per-axes legends with one legend to the right of the last axes, clear of the bars."""
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    axes[-1].legend(handles, labels, title="", loc="center left", frameon=False, bbox_to_anchor=(1.02, 0.5))


def topology_comparison(trials: pd.DataFrame, out_dir: Path) -> None:
    """Mesh vs hub-and-spoke on the shared conditions: failure rate and rounds (LLM)."""
    matched = _add_condition(topology_matched(trials[trials["agent_type"] == "llm"]))
    matched["failure_rate_pct"] = 100 * matched["failed"]
    matched["topology"] = matched["topology"].map(TOPOLOGY_LABELS)
    shared = {"x": "condition", "hue": "topology", "order": _condition_order(matched),
              "hue_order": list(TOPOLOGY_LABELS.values())}

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.barplot(data=matched, y="failure_rate_pct", errorbar=None, ax=left, **shared)
    left.set(xlabel="Latency / packet loss", ylabel="Failure rate (%)", ylim=(0, 100), title="Failure rate")
    sns.barplot(
        data=matched, y="rounds_to_consensus", errorbar=("ci", 95), seed=SEED, capsize=0.1, ax=right, **shared
    )
    right.set(
        xlabel="Latency / packet loss",
        ylabel=f"Mean rounds (failures count as {MAX_ROUNDS})",
        ylim=(0, MAX_ROUNDS + 1),
        title="Rounds to consensus (mean, 95% CI)",
    )
    _shared_legend([left, right])
    fig.suptitle("Topology comparison (LLM agent)")
    fig.tight_layout()
    _save(fig, out_dir, "topology_comparison")


def llm_vs_baseline(trials: pd.DataFrame, out_dir: Path) -> None:
    """Convergence rate and rounds per mesh condition for both agent types."""
    mesh = _add_condition(trials[trials["topology"] == "mesh"])
    mesh["convergence_rate_pct"] = 100 * mesh["converged"]
    mesh["agent_type"] = mesh["agent_type"].map(AGENT_LABELS)
    shared = {"x": "condition", "hue": "agent_type", "order": _condition_order(mesh),
              "hue_order": list(AGENT_LABELS.values())}

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    sns.barplot(data=mesh, y="convergence_rate_pct", errorbar=None, ax=top, **shared)
    top.set(xlabel="", ylabel="Convergence rate (%)", ylim=(0, 100), title="Convergence rate")
    sns.barplot(
        data=mesh.dropna(subset=["rounds_converged"]),
        y="rounds_converged", errorbar=("ci", 95), seed=SEED, capsize=0.1, ax=bottom, **shared,
    )
    bottom.set(
        xlabel="Latency / packet loss",
        ylabel="Rounds to consensus",
        ylim=(0, MAX_ROUNDS + 1),
        title="Rounds to consensus (converged trials; mean, 95% CI)",
    )
    _shared_legend([top, bottom])
    fig.suptitle("LLM vs baseline (mesh)")
    fig.tight_layout()
    _save(fig, out_dir, "llm_vs_baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Phase 4 figures.")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    trials = load_trials(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failure_heatmap(trials, out_dir)
    rounds_vs_loss(trials, out_dir)
    topology_comparison(trials, out_dir)
    llm_vs_baseline(trials, out_dir)


if __name__ == "__main__":
    main()