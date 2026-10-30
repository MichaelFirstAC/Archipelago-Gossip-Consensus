"""Gossip protocol driver for round-based push gossip communication."""

import asyncio
import inspect
import random
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.agents import belief as bel
from src.agents.baseline_agent import BaselineAgent
from src.agents.schema import Action
from src.logging.run_logger import RunLogger
from src.network.link import Link
from src.network.node import Node
from src.simulation.consensus import DEFAULT_EPSILON, check_consensus


class GossipProtocol:
    """Orchestrates push-gossip rounds between nodes across a simulated link.

    Each round: nodes push their belief vector to random neighbors through the
    link and wait at most ``round_ms`` for delivery. A message slower than the
    round window keeps flying and lands in the receiver's inbox during a later
    round, so it is used late and carries stale beliefs; this is how latency
    reaches consensus time. Every node then applies the DeGroot update with the
    messages that actually arrived (dropped ones never do), and its agent decides
    an action and vote from the updated state.
    """

    def __init__(
        self,
        nodes: List[Node],
        link: Link,
        fanout: int = 1,
        epsilon: float = DEFAULT_EPSILON,
        logger: Optional[RunLogger] = None,
        agent_type: str = "dummy",
        round_ms: Optional[float] = 1000.0,
    ):
        self.nodes = nodes
        self.node_map: Dict[str, Node] = {n.id: n for n in nodes}
        self.candidates: List[str] = [n.id for n in nodes]
        self.link = link
        self.fanout = fanout
        self.epsilon = epsilon
        self.logger = logger
        self.agent_type = agent_type
        self.round_ms = round_ms
        self._in_flight: Set[asyncio.Task] = set()
        self.total_messages_sent = 0
        self.parse_failures = 0
        self._references: Dict[str, BaselineAgent] = {
            n.id: n.agent if isinstance(n.agent, BaselineAgent) else BaselineAgent(n.id, self.candidates, n.archetype)
            for n in nodes
            if n.archetype is not None
        }

    async def execute_round(self, round_num: int) -> bool:
        """Run a single gossip round across all nodes.

        1. Each node pushes its belief to `fanout` randomly selected neighbors.
        2. The round waits up to `round_ms` for the Link to deliver (or drop) them;
           slower messages arrive during a later round.
        3. Each node updates its belief from what arrived, then its agent decides.
        4. Consensus is checked.

        Returns:
            True if consensus is reached at the end of this round, False otherwise.
        """
        send_tasks: List[asyncio.Task] = []

        for node in self.nodes:
            if not node.neighbors:
                continue

            k = min(self.fanout, len(node.neighbors))
            for neighbor_id in random.sample(node.neighbors, k):
                message = {
                    "sender": node.id,
                    "round": round_num,
                    "payload": {
                        "node_id": node.id,
                        "faction_name": node.faction_name,
                        "belief": node.belief,
                        "belief_vector": None if node.belief_vector is None else node.belief_vector.tolist(),
                    },
                }
                task = asyncio.create_task(
                    self.link.send(
                        round_num=round_num,
                        sender_id=node.id,
                        receiver_node=self.node_map[neighbor_id],
                        message=message,
                    )
                )
                self._in_flight.add(task)
                task.add_done_callback(self._in_flight.discard)
                send_tasks.append(task)

        if send_tasks:
            timeout = None if self.round_ms is None else self.round_ms / 1000.0
            await asyncio.wait(send_tasks, timeout=timeout)
            self.total_messages_sent += len(send_tasks)

        for node in self.nodes:
            inbox = node.drain_inbox()
            self._update_belief(node, inbox)
            if node.agent is not None:
                await self._run_agent(node, inbox, round_num)

        converged, _ = check_consensus(self.nodes, self.epsilon)
        return converged

    def _update_belief(self, node: Node, inbox: List[Dict[str, Any]]) -> None:
        """Apply the DeGroot step using only the belief vectors that arrived this round."""
        if node.belief_vector is None:
            return
        received = [
            (m["sender"], np.asarray(m["payload"]["belief_vector"], dtype=float))
            for m in inbox
            if m["payload"].get("belief_vector") is not None
        ]
        openness = node.archetype.openness if node.archetype is not None else 0.5
        node.belief_vector = bel.degroot_update(node.belief_vector, received, openness)
        node.belief = self.candidates[bel.top_candidate(node.belief_vector)]

    async def _run_agent(self, node: Node, inbox: List[Dict[str, Any]], round_num: int) -> None:
        """Ask the node's agent for an action, log the decision, and apply it."""
        state = {**node.get_state(), "neighbors": node.neighbors, "candidates": self.candidates}
        baseline_vote = self._baseline_vote(node, state, inbox)

        started = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(node.agent.decide):
                action = await node.agent.decide(state, inbox)
            else:
                action = node.agent.decide(state, inbox)
            valid_json = True
        except Exception as e:
            self.parse_failures += 1
            valid_json = False
            action = Action(action="hold", target=None, vote=node.vote, dialogue=f"Fallback error: {e}")
        latency_ms = (time.perf_counter() - started) * 1000.0

        valid_action = self._is_legal(action)
        if self.logger is not None:
            self.logger.log_decision(
                round_num=round_num,
                node=node.id,
                agent_type=self.agent_type,
                valid_json=valid_json,
                valid_action=valid_action,
                vote=action.vote,
                baseline_vote=baseline_vote,
                decision_latency_ms=latency_ms,
                tokens=getattr(node.agent, "last_tokens", 0),
            )

        if valid_action:
            node.vote = action.vote

    def _baseline_vote(self, node: Node, state: Dict[str, Any], inbox: List[Dict[str, Any]]) -> Optional[str]:
        """What the baseline agent would vote for from this exact state (None without an archetype)."""
        reference = self._references.get(node.id)
        return None if reference is None else reference.decide(state, inbox).vote

    def _is_legal(self, action: Action) -> bool:
        """An action is legal when its vote is a known candidate and its target (if any) a known faction."""
        return action.vote in self.candidates and (action.target is None or action.target in self.node_map)

    async def run(self, max_rounds: int = 15) -> Dict[str, Any]:
        """Drive gossip rounds until consensus or max_rounds is reached.

        Returns:
            Dictionary summary with keys 'converged', 'rounds_to_consensus',
            'total_messages', 'parse_failures' and 'consensus_value'.
        """
        converged, consensus_value = check_consensus(self.nodes, self.epsilon)
        if converged:
            return {
                "converged": True,
                "rounds_to_consensus": 0,
                "total_messages": 0,
                "parse_failures": 0,
                "consensus_value": consensus_value,
            }

        rounds_completed = 0
        try:
            for r in range(1, max_rounds + 1):
                rounds_completed = r
                if await self.execute_round(r):
                    break
        finally:
            for task in list(self._in_flight):
                task.cancel()
            await asyncio.gather(*self._in_flight, return_exceptions=True)

        converged, consensus_value = check_consensus(self.nodes, self.epsilon)
        return {
            "converged": converged,
            "rounds_to_consensus": rounds_completed,
            "total_messages": self.total_messages_sent,
            "parse_failures": self.parse_failures,
            "consensus_value": consensus_value,
        }
