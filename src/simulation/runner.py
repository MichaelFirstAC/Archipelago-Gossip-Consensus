"""Single-run orchestrator for a single trial execution."""

import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.dummy_agent import DummyAgent
from src.logging.run_logger import RunLogger
from src.network.gossip import GossipProtocol
from src.network.link import Link
from src.network.node import Node
from src.network.topology import build_hub_spoke, build_mesh


async def run_trial(
    trial_id: str,
    latency_ms: float = 0.0,
    loss_rate: float = 0.0,
    topology: str = "mesh",
    num_agents: int = 5,
    max_rounds: int = 15,
    archetype_mix: Optional[Dict[str, float]] = None,
    agent_type: str = "dummy",
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
        archetype_mix: Distribution of archetypes (for LLM agents).
        agent_type: 'dummy' or 'llm'.
        log_dir: Directory where event and summary CSV logs will be saved.
        save_logs: Whether to persist CSV logs to disk.
        master_summary_filename: Filename for the accumulated summary log.

    Returns:
        Summary dictionary with convergence, rounds, messages, and failures.
    """
    # 1. Topology adjacency construction
    node_ids = [f"faction_{i}" for i in range(num_agents)]
    if topology == "mesh":
        adjacency = build_mesh(node_ids)
    elif topology == "hub_spoke":
        adjacency = build_hub_spoke(node_ids, hub_id=node_ids[0])
    else:
        raise ValueError(f"Unknown topology: {topology}. Must be 'mesh' or 'hub_spoke'.")

    # 2. Node & Agent instantiation
    nodes: List[Node] = []
    for nid in node_ids:
        if agent_type == "dummy":
            agent = DummyAgent(mode="scripted", node_id=nid)
        elif agent_type == "llm":
            # Deferred to Phase 2
            from src.agents.llm_agent import LLMAgent  # type: ignore
            agent = LLMAgent(node_id=nid)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

        node = Node(
            node_id=nid,
            neighbors=adjacency.get(nid, []),
            initial_belief=nid,
            agent=agent,
        )
        nodes.append(node)

    # 3. Logger and Link setup
    logger = RunLogger(trial_id=trial_id) if save_logs else None
    link = Link(latency_ms=latency_ms, loss_rate=loss_rate, logger=logger)

    # 4. Drive gossip protocol rounds
    protocol = GossipProtocol(nodes=nodes, link=link, fanout=1)
    result = await protocol.run(max_rounds=max_rounds)

    # 5. Format trial summary
    archetype_repr = (
        ",".join(f"{k}:{v}" for k, v in archetype_mix.items())
        if archetype_mix
        else "scripted_dummy"
    )

    summary: Dict[str, Any] = {
        "trial_id": trial_id,
        "latency": latency_ms,
        "loss_rate": loss_rate,
        "topology": topology,
        "archetype_mix": archetype_repr,
        "converged": result["converged"],
        "rounds_to_consensus": result["rounds_to_consensus"],
        "total_messages": result["total_messages"],
        "parse_failures": result["parse_failures"],
    }

    # 6. Save logs to disk
    if save_logs and logger is not None:
        out_dir = Path(log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        event_path = out_dir / f"events_{trial_id}.csv"
        logger.write_events_csv(event_path)

        master_summary_path = out_dir / master_summary_filename
        RunLogger.append_summary_csv(master_summary_path, summary)

    return summary


def main():
    """CLI entrypoint for running a single trial."""
    parser = argparse.ArgumentParser(description="Run a single Archipelago consensus trial.")
    parser.add_argument("--trial-id", type=str, default="test_trial_001")
    parser.add_argument("--latency", type=float, default=0.0)
    parser.add_argument("--loss", type=float, default=0.0)
    parser.add_argument("--topology", type=str, default="mesh", choices=["mesh", "hub_spoke"])
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=15)
    parser.add_argument("--agent-type", type=str, default="dummy", choices=["dummy", "llm"])
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
            agent_type=args.agent_type,
            log_dir=args.log_dir,
        )
    )
    print("Trial completed:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
