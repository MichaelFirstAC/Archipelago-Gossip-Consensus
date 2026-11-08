"""Loads all run CSVs into analysis-ready dataframes.

Typical use:
    trials, decisions = load_all("data/runs")

``trials`` has one row per trial; ``decisions`` has one row per agent decision
with the trial's condition columns attached.
"""

import re
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

PathLike = Union[str, Path]

DEFAULT_RUNS_DIR = "data/runs"
FACTION_ID = r"faction_\d+"
RUN_PATTERN = re.compile(r"_run(\d+)$")
CONDITION_COLUMNS = ["agent_type", "topology", "latency", "loss_rate"]


def load_trials(runs_dir: PathLike = DEFAULT_RUNS_DIR) -> pd.DataFrame:
    """Read ``trials_summary.csv`` and add parsed and helper columns.

    Added columns:
        run: run index parsed from ``trial_id``.
        failed: True when consensus was not reached (``rounds_to_consensus`` is
            then the 20-round cap, i.e. right-censored, not a real consensus time).
        rounds_converged: ``rounds_to_consensus`` for converged trials, NaN otherwise.
    """
    trials = pd.read_csv(Path(runs_dir) / "trials_summary.csv")
    trials["converged"] = trials["converged"].astype(bool)
    trials["run"] = trials["trial_id"].str.extract(RUN_PATTERN, expand=False).astype(int)
    trials["failed"] = ~trials["converged"]
    trials["rounds_converged"] = trials["rounds_to_consensus"].where(trials["converged"])
    return trials


def load_decisions(trials: pd.DataFrame, runs_dir: PathLike = DEFAULT_RUNS_DIR) -> pd.DataFrame:
    """Concatenate every ``decisions_<trial_id>.csv`` and attach condition columns.

    Added columns:
        vote_is_faction: True when ``vote`` is a real faction id (invalid LLM
            actions can leave junk such as ',').
        agrees_with_baseline: True when ``vote`` equals ``baseline_vote``; NaN
            where the vote is not a faction id, so averages exclude junk votes.

    Raises:
        FileNotFoundError: if a trial listed in the summary has no decision log.
    """
    runs_dir = Path(runs_dir)
    frames = []
    for trial_id in trials["trial_id"]:
        path = runs_dir / f"decisions_{trial_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing decision log for trial {trial_id}: {path}")
        frames.append(pd.read_csv(path, dtype={"vote": str, "baseline_vote": str}))

    decisions = pd.concat(frames, ignore_index=True)
    decisions = decisions.drop(columns="agent_type").merge(
        trials[["trial_id", *CONDITION_COLUMNS]], on="trial_id", how="left"
    )
    decisions["vote_is_faction"] = decisions["vote"].str.fullmatch(FACTION_ID).fillna(False)
    agrees = decisions["vote"] == decisions["baseline_vote"]
    decisions["agrees_with_baseline"] = agrees.where(decisions["vote_is_faction"])
    return decisions


def load_all(runs_dir: PathLike = DEFAULT_RUNS_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(trials, decisions)`` for every trial logged in ``runs_dir``."""
    trials = load_trials(runs_dir)
    return trials, load_decisions(trials, runs_dir)


def llm_mesh(trials: pd.DataFrame) -> pd.DataFrame:
    """Subset used for the Research Methods ANOVA: LLM arm, mesh topology."""
    return trials[(trials["agent_type"] == "llm") & (trials["topology"] == "mesh")]


def topology_matched(llm_trials: pd.DataFrame) -> pd.DataFrame:
    """LLM trials on the (latency, loss) conditions that ran under both topologies."""
    shared = llm_trials.groupby(["latency", "loss_rate"])["topology"].nunique()
    keep = shared[shared == 2].index
    return llm_trials.set_index(["latency", "loss_rate"]).loc[keep].reset_index()


if __name__ == "__main__":
    trial_df, decision_df = load_all()
    print(f"trials: {len(trial_df)} rows | decisions: {len(decision_df)} rows")
    print(trial_df.groupby(["agent_type", "topology"]).size())