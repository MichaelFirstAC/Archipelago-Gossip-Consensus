"""Statistical tests: Two-Way ANOVA, failure-rate tests, Mann-Whitney U and AI metrics.

Research Methods analysis (ANOVA, failure rate, topology) uses the LLM arm only.
The baseline arm appears only in the LLM-vs-baseline comparison (AI course).

Run from the repo root:
    python -m analysis.stats [--runs-dir data/runs] [--out-dir results/tables]

Every table is printed and saved as a CSV in ``--out-dir``.
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from analysis.load_logs import DEFAULT_RUNS_DIR, llm_mesh, load_all, topology_matched

DEFAULT_OUT_DIR = "results/tables"
CONDITION = ["agent_type", "topology", "latency", "loss_rate"]
ANOVA_LABELS = {
    "C(latency)": "latency",
    "C(loss_rate)": "loss_rate",
    "C(latency):C(loss_rate)": "latency x loss_rate",
}


def _fisher(fail_a: int, n_a: int, fail_b: int, n_b: int) -> Dict[str, float]:
    """Fisher exact test on failures vs successes for two groups (odds ratio of a vs b)."""
    odds_ratio, p_value = stats.fisher_exact([[fail_a, n_a - fail_a], [fail_b, n_b - fail_b]])
    return {"odds_ratio": odds_ratio, "p_value": p_value}


def _mann_whitney(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    """Two-sided Mann-Whitney U for x vs y, with rank-biserial effect size.

    A positive ``rank_biserial`` means x tends to be larger than y (range -1 to 1).
    """
    u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {
        "u_statistic": u_stat,
        "p_value": p_value,
        "rank_biserial": 2 * u_stat / (len(x) * len(y)) - 1,
    }


def condition_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Per-condition convergence, failure rate and rounds, plus effect vs the control cell.

    Rounds statistics use converged trials only (failures are capped at 20 rounds).
    The control is latency=0, loss=0 within the same agent type and topology;
    ``failure_diff_vs_control`` is in percentage points, with a Fisher exact p-value.
    """
    rows = []
    for (agent, topology, latency, loss), group in trials.groupby(CONDITION):
        rows.append(
            {
                "agent_type": agent,
                "topology": topology,
                "latency": latency,
                "loss_rate": loss,
                "n_trials": len(group),
                "n_failed": int(group["failed"].sum()),
                "convergence_rate": group["converged"].mean(),
                "failure_rate": group["failed"].mean(),
                "mean_rounds_converged": group["rounds_converged"].mean(),
                "median_rounds_converged": group["rounds_converged"].median(),
            }
        )
    table = pd.DataFrame(rows)

    diffs, p_values = [], []
    for _, row in table.iterrows():
        control = table[
            (table["agent_type"] == row["agent_type"])
            & (table["topology"] == row["topology"])
            & (table["latency"] == 0)
            & (table["loss_rate"] == 0)
        ].iloc[0]
        diffs.append(100 * (row["failure_rate"] - control["failure_rate"]))
        fisher = _fisher(row["n_failed"], row["n_trials"], control["n_failed"], control["n_trials"])
        p_values.append(fisher["p_value"])
    table["failure_diff_vs_control"] = diffs
    table["p_vs_control"] = p_values
    return table


def anova_rounds(llm_mesh_trials: pd.DataFrame) -> pd.DataFrame:
    """Two-Way ANOVA: rounds_to_consensus ~ latency * loss_rate (LLM arm, mesh).

    Only converged trials are used, because failures are capped at 20 rounds and
    are not real consensus times; the failure-rate tests cover them instead.
    Latency and loss are treated as categorical levels. Cell sizes are unequal
    (failures drop out), so Type II sums of squares are used.
    """
    converged = llm_mesh_trials[llm_mesh_trials["converged"]]
    model = ols("rounds_to_consensus ~ C(latency) * C(loss_rate)", data=converged).fit()
    table = anova_lm(model, typ=2)
    residual_ss = table.loc["Residual", "sum_sq"]
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + residual_ss)
    table.loc["Residual", "partial_eta_sq"] = np.nan
    return table.rename(index=ANOVA_LABELS).reset_index(names="effect")


def failure_rate_tests(llm_mesh_trials: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test of failure vs each factor (LLM arm, mesh), with Cramer's V."""
    rows = []
    for factor in ["latency", "loss_rate"]:
        counts = pd.crosstab(llm_mesh_trials[factor], llm_mesh_trials["failed"])
        chi2, p_value, dof, _ = stats.chi2_contingency(counts)
        rows.append(
            {
                "factor": factor,
                "chi2": chi2,
                "dof": dof,
                "p_value": p_value,
                "cramers_v": np.sqrt(chi2 / counts.to_numpy().sum()),
            }
        )
    return pd.DataFrame(rows)


def topology_test(llm_trials: pd.DataFrame) -> pd.DataFrame:
    """Hub-and-spoke vs mesh on the conditions both topologies share (LLM arm).

    Rounds with failures counted at the 20-round cap (a failure is the worst
    outcome, so it ranks last), rounds among converged trials only, and the
    failure rate. ``rank_biserial`` > 0 means hub_spoke is slower than mesh.
    """
    matched = topology_matched(llm_trials)
    hub = matched[matched["topology"] == "hub_spoke"]
    mesh = matched[matched["topology"] == "mesh"]

    all_rounds = _mann_whitney(hub["rounds_to_consensus"], mesh["rounds_to_consensus"])
    converged = _mann_whitney(hub["rounds_converged"].dropna(), mesh["rounds_converged"].dropna())
    failure = _fisher(int(hub["failed"].sum()), len(hub), int(mesh["failed"].sum()), len(mesh))
    return pd.DataFrame(
        [
            {"metric": "rounds (failures = cap)", "n_hub": len(hub), "n_mesh": len(mesh), **all_rounds},
            {
                "metric": "rounds (converged only)",
                "n_hub": hub["converged"].sum(),
                "n_mesh": mesh["converged"].sum(),
                **converged,
            },
            {
                "metric": "failure rate (odds ratio, hub vs mesh)",
                "n_hub": len(hub),
                "n_mesh": len(mesh),
                **failure,
            },
        ]
    )


def llm_vs_baseline(trials: pd.DataFrame) -> pd.DataFrame:
    """LLM vs baseline on the mesh conditions both arms ran (AI-course comparison)."""
    mesh = trials[trials["topology"] == "mesh"]
    llm = mesh[mesh["agent_type"] == "llm"]
    baseline = mesh[mesh["agent_type"] == "baseline"]

    failure = _fisher(int(llm["failed"].sum()), len(llm), int(baseline["failed"].sum()), len(baseline))
    all_rounds = _mann_whitney(llm["rounds_to_consensus"], baseline["rounds_to_consensus"])
    converged = _mann_whitney(llm["rounds_converged"].dropna(), baseline["rounds_converged"].dropna())
    return pd.DataFrame(
        [
            {
                "metric": "failure rate (odds ratio, llm vs baseline)",
                "llm": llm["failed"].mean(),
                "baseline": baseline["failed"].mean(),
                **failure,
            },
            {
                "metric": "rounds (failures = cap)",
                "llm": llm["rounds_to_consensus"].median(),
                "baseline": baseline["rounds_to_consensus"].median(),
                **all_rounds,
            },
            {
                "metric": "rounds (converged only)",
                "llm": llm["rounds_converged"].median(),
                "baseline": baseline["rounds_converged"].median(),
                **converged,
            },
        ]
    )


def ai_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    """AI metrics per agent type (SPEC section 5.5).

    Action validity is the share of parsed (valid-JSON) outputs that were legal.
    Agreement with the baseline vote is given over decisions with a real faction
    vote, and over all decisions for comparison; it is only meaningful for the LLM.
    Latency and tokens are reported as mean and median because the first LLM call
    of a run is a slow warm-up outlier.
    """
    rows = []
    for agent, group in decisions.groupby("agent_type"):
        parsed = group[group["valid_json"]]
        is_llm = agent == "llm"
        rows.append(
            {
                "agent_type": agent,
                "n_decisions": len(group),
                "json_validity": group["valid_json"].mean(),
                "action_validity": parsed["valid_action"].mean(),
                "agreement_valid_votes": group["agrees_with_baseline"].mean() if is_llm else np.nan,
                "agreement_all_decisions": (group["vote"] == group["baseline_vote"]).mean() if is_llm else np.nan,
                "latency_ms_mean": group["decision_latency_ms"].mean(),
                "latency_ms_median": group["decision_latency_ms"].median(),
                "tokens_mean": group["tokens"].mean(),
                "tokens_median": group["tokens"].median(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 4 statistical analysis.")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    trials, decisions = load_all(args.runs_dir)
    results = {
        "condition_summary": condition_table(trials),
        "anova_rounds_llm_mesh": anova_rounds(llm_mesh(trials)),
        "failure_rate_tests_llm_mesh": failure_rate_tests(llm_mesh(trials)),
        "topology_llm": topology_test(trials[trials["agent_type"] == "llm"]),
        "llm_vs_baseline_mesh": llm_vs_baseline(trials),
        "ai_metrics": ai_metrics(decisions),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(table.to_string(index=False, float_format="{:.4g}".format))
    print(f"\nSaved {len(results)} tables to {out_dir}/")


if __name__ == "__main__":
    main()