"""Gossip protocol driver for round-based push gossip communication."""

import asyncio
import inspect
import random
from typing import Any, Dict, List, Optional, Tuple

from src.network.link import Link
from src.network.node import Node
from src.simulation.consensus import check_consensus


class GossipProtocol:
    """Orchestrates push-gossip rounds between nodes across a simulated link."""

    def __init__(self, nodes: List[Node], link: Link, fanout: int = 1):
        self.nodes = nodes
        self.node_map: Dict[str, Node] = {n.id: n for n in nodes}
        self.link = link
        self.fanout = fanout
        self.total_messages_sent = 0
        self.parse_failures = 0

    async def execute_round(self, round_num: int) -> bool:
        """Run a single gossip round across all nodes.

        1. Each node pushes its state to `fanout` randomly selected neighbors.
        2. In-flight messages are delivered (or dropped) by the Link.
        3. Recipient nodes drain their inbox, consult their agent, and update belief.
        4. Consensus is checked.

        Returns:
            True if consensus is reached at the end of this round, False otherwise.
        """
        send_tasks = []

        for node in self.nodes:
            if not node.neighbors:
                continue

            # Pick random neighbor(s) according to fanout
            k = min(self.fanout, len(node.neighbors))
            chosen_neighbor_ids = random.sample(node.neighbors, k)

            for neighbor_id in chosen_neighbor_ids:
                neighbor_node = self.node_map.get(neighbor_id)
                if neighbor_node is None:
                    continue

                message = {
                    "sender": node.id,
                    "round": round_num,
                    "payload": {
                        "node_id": node.id,
                        "faction_name": node.faction_name,
                        "belief": node.belief,
                    },
                }

                send_tasks.append(
                    self.link.send(
                        round_num=round_num,
                        sender_id=node.id,
                        receiver_node=neighbor_node,
                        message=message,
                    )
                )

        if send_tasks:
            await asyncio.gather(*send_tasks)
            self.total_messages_sent += len(send_tasks)

        # Agent decision phase: nodes drain inboxes and update belief state
        for node in self.nodes:
            inbox_messages = node.drain_inbox()
            if node.agent is None:
                continue

            # Support both synchronous and asynchronous decide() methods
            decide_fn = node.agent.decide
            try:
                if inspect.iscoroutinefunction(decide_fn):
                    action_data = await decide_fn(
                        node_id=node.id,
                        current_belief=node.belief,
                        inbox_messages=inbox_messages,
                        known_neighbors=node.neighbors,
                    )
                else:
                    action_data = decide_fn(
                        node_id=node.id,
                        current_belief=node.belief,
                        inbox_messages=inbox_messages,
                        known_neighbors=node.neighbors,
                    )
            except Exception as e:
                self.parse_failures += 1
                action_data = {"action": "hold", "target": None, "dialogue": f"Fallback error: {e}"}

            self._apply_action(node, action_data)

        converged, _ = check_consensus(self.nodes)
        return converged

    def _apply_action(self, node: Node, action_data: Dict[str, Any]) -> None:
        """Apply the chosen action to the node's internal state."""
        action = action_data.get("action", "hold")
        target = action_data.get("target")

        if action == "forge_alliance" and target:
            node.belief = str(target)
        elif action == "break_alliance":
            node.belief = node.id
        # 'hold' and 'broadcast_state' retain current belief

    async def run(self, max_rounds: int = 15) -> Dict[str, Any]:
        """Drive gossip rounds until consensus or max_rounds is reached.

        Returns:
            Dictionary summary:
            {
                'converged': bool,
                'rounds_to_consensus': int,
                'total_messages': int,
                'parse_failures': int,
                'consensus_value': Optional[str]
            }
        """
        converged, consensus_value = check_consensus(self.nodes)
        if converged:
            return {
                "converged": True,
                "rounds_to_consensus": 0,
                "total_messages": 0,
                "parse_failures": 0,
                "consensus_value": consensus_value,
            }

        rounds_completed = 0
        for r in range(1, max_rounds + 1):
            rounds_completed = r
            converged = await self.execute_round(r)
            if converged:
                _, consensus_value = check_consensus(self.nodes)
                return {
                    "converged": True,
                    "rounds_to_consensus": rounds_completed,
                    "total_messages": self.total_messages_sent,
                    "parse_failures": self.parse_failures,
                    "consensus_value": consensus_value,
                }

        _, consensus_value = check_consensus(self.nodes)
        return {
            "converged": False,
            "rounds_to_consensus": rounds_completed,
            "total_messages": self.total_messages_sent,
            "parse_failures": self.parse_failures,
            "consensus_value": consensus_value,
        }
