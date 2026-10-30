"""Single-run orchestrator for a single trial execution."""

import argparse
import asyncio
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.agents import belief as bel
from src.agents.archetypes import DEFAULT_MIX, assign_archetypes
from src.agents.factory import AGENT_TYPES, create_agent
from src.logging.run_logger import RunLogger
from src.network.gossip import GossipProtocol
from src.network.link import Link
from src.network.node import Node
from src.network.topology import build_hub_spoke, build_mesh
from src.simulation.consensus import DEFAULT_EPSILON


async def run_trial(
    trial_id: str,
    latency_ms: float = 0.0,
    loss_rate: float = 0.0,
    topology: str = "mesh",
    num_agents: int = 5,
    max_rounds: int = 20,
    fanout: int = 1,
    round_ms: Optional[float] = 1000.0,
    archetype_mix: Optional[Dict[str, float]] = None,
    agent_type: str = "dummy",
    epsilon: float = DEFAULT_EPSILON,
    seed: Optional[int] = None,
    log_dir: str = "data/runs",
    save_logs: bool = True,
    master_summary_filename: str = "trials_summary.csv",
) -> Dict[str, Any]:
    """Execute a single simulation trial under specified conditions.

    Args:
        trial_id: Unique identifier for the trial.
        latency_ms: Link latency in milliseconds.
        loss_rate: Packet loss probability [0.0, 1.0].
        topology: 'mesh' or 'hub_spoke'.
        num_agents: Number of faction nodes.
        max_rounds: Maximum gossip rounds to run.
        fanout: Random neighbors each node pushes to per round.
        round_ms: Gossip period. Messages slower than this arrive in later rounds (stale).
        archetype_mix: Archetype proportions; defaults to the SPEC mix.
        agent_type: 'dummy', 'baseline' or 'llm'.
        epsilon: Consensus threshold on the belief consensus score.
        seed: Seeds archetype assignment, initial beliefs, gossip partner choice and packet drops.
        log_dir: Directory where CSV logs will be saved.
        save_logs: Whether to persist CSV logs to disk.
        master_summary_filename: Filename for the accumulated summary log.

    Returns:
        Summary dictionary with convergence, rounds, messages, and failures.
    """
    mix = archetype_mix or DEFAULT_MIX
    rng = np.random.default_rng(seed)
    if seed is not None:
        random.seed(seed)

    # 1. Topology adjacency construction
    node_ids = [f"faction_{i}" for i in range(num_agents)]
    if topology == "mesh":
        adjacency = build_mesh(node_ids)
    elif topology == "hub_spoke":
        adjacency = build_hub_spoke(node_ids, hub_id=node_ids[0])
    else:
        raise ValueError(f"Unknown topology: {topology}. Must be 'mesh' or 'hub_spoke'.")

    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unknown agent_type: {agent_type!r}. Expected one of {AGENT_TYPES}.")

    # 2. Node, archetype, belief and agent instantiation
    archetypes = assign_archetypes(node_ids, mix, rng)
    nodes: List[Node] = []
    for i, nid in enumerate(node_ids):
        vector = bel.initial_belief(num_agents, i, rng)
        nodes.append(
            Node(
                node_id=nid,
                neighbors=adjacency.get(nid, []),
                initial_belief=node_ids[bel.top_candidate(vector)],
                agent=create_agent(agent_type, nid, node_ids, archetypes[nid]),
                belief_vector=vector,
                archetype=archetypes[nid],
            )
        )

    # 3. Logger and Link setup
    logger = RunLogger(trial_id=trial_id) if save_logs else None
    link = Link(latency_ms=latency_ms, loss_rate=loss_rate, logger=logger)

    # 4. Drive gossip protocol rounds
    protocol = GossipProtocol(
        nodes=nodes, link=link, fanout=fanout, epsilon=epsilon, logger=logger,
        agent_type=agent_type, round_ms=round_ms,
    )
    result = await protocol.run(max_rounds=max_rounds)

    # 5. Format trial summary
    summary: Dict[str, Any] = {
        "trial_id": trial_id,
        "agent_type": agent_type,
        "latency": latency_ms,
        "loss_rate": loss_rate,
        "topology": topology,
        "archetype_mix": ",".join(f"{k}:{v}" for k, v in mix.items()),
        "converged": result["converged"],
        "rounds_to_consensus": result["rounds_to_consensus"],
        "total_messages": result["total_messages"],
        "parse_failures": result["parse_failures"],
    }

    # 6. Save logs to disk
    if save_logs and logger is not None:
        out_dir = Path(log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.write_events_csv(out_dir / f"events_{trial_id}.csv")
        logger.write_decisions_csv(out_dir / f"decisions_{trial_id}.csv")
        RunLogger.append_summary_csv(out_dir / master_summary_filename, summary)

    return summary


def main():
    """CLI entrypoint for running a single trial."""
    parser = argparse.ArgumentParser(description="Run a single Archipelago consensus trial.")
    parser.add_argument("--trial-id", type=str, default="test_trial_001")
    parser.add_argument("--latency", type=float, default=0.0)
    parser.add_argument("--loss", type=float, default=0.0)
    parser.add_argument("--topology", type=str, default="mesh", choices=["mesh", "hub_spoke"])
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--fanout", type=int, default=1)
    parser.add_argument("--round-ms", type=float, default=1000.0)
    parser.add_argument("--agent-type", type=str, default="baseline", choices=list(AGENT_TYPES))
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="data/runs")

    args = parser.parse_args()

    result = asyncio.run(
        run_trial(
            trial_id=args.trial_id,
            latency_ms=args.latency,
            loss_rate=args.loss,
            topology=args.topology,
            num_agents=args.agents,
            max_rounds=args.max_rounds,
            fanout=args.fanout,
            round_ms=args.round_ms,
            agent_type=args.agent_type,
            epsilon=args.epsilon,
            seed=args.seed,
            log_dir=args.log_dir,
        )
    )
    print("Trial completed:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
