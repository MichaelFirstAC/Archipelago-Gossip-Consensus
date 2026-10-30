"""Verifies the round deadline: messages slower than the gossip period arrive in later rounds."""

import asyncio
import sys
from pathlib import Path

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.archetypes import ARCHETYPES
from src.network.gossip import GossipProtocol
from src.network.link import Link
from src.network.node import Node
from src.network.topology import build_mesh


def _two_nodes():
    adjacency = build_mesh(["faction_0", "faction_1"])
    return [
        Node("faction_0", neighbors=adjacency["faction_0"], initial_belief="faction_0",
             belief_vector=np.array([1.0, 0.0]), archetype=ARCHETYPES["cooperative"]),
        Node("faction_1", neighbors=adjacency["faction_1"], initial_belief="faction_1",
             belief_vector=np.array([0.0, 1.0]), archetype=ARCHETYPES["cooperative"]),
    ]


async def test_slow_message_arrives_late():
    print("[TEST] A message slower than the round window is applied in a later round...")
    nodes = _two_nodes()
    # 250 ms latency vs a 100 ms round window: round 1 must finish with nothing delivered.
    protocol = GossipProtocol(nodes, Link(latency_ms=250, loss_rate=0.0), round_ms=100)
    await protocol.execute_round(1)
    assert np.allclose(nodes[0].belief_vector, [1.0, 0.0]), "belief changed although nothing arrived in time"
    assert np.allclose(nodes[1].belief_vector, [0.0, 1.0])

    await asyncio.sleep(0.3)  # the in-flight messages now land in the inboxes
    await protocol.execute_round(2)
    assert nodes[0].belief_vector[1] > 0.0, "the late round-1 message was never used"

    for task in list(protocol._in_flight):
        task.cancel()
    await asyncio.gather(*protocol._in_flight, return_exceptions=True)
    print("  -> OK")


async def test_fast_message_arrives_in_round():
    print("[TEST] A message faster than the round window is applied in the same round...")
    nodes = _two_nodes()
    protocol = GossipProtocol(nodes, Link(latency_ms=10, loss_rate=0.0), round_ms=100)
    await protocol.execute_round(1)
    assert nodes[0].belief_vector[1] > 0.0 and nodes[1].belief_vector[0] > 0.0
    print("  -> OK")


async def test_run_cancels_in_flight():
    print("[TEST] Messages still in flight when a run ends are cancelled...")
    nodes = _two_nodes()
    protocol = GossipProtocol(nodes, Link(latency_ms=5000, loss_rate=0.0), round_ms=20)
    await protocol.run(max_rounds=2)
    assert not protocol._in_flight, "leftover in-flight tasks"
    print("  -> OK")


def main():
    async def run_all():
        await test_slow_message_arrives_late()
        await test_fast_message_arrives_in_round()
        await test_run_cancels_in_flight()

    asyncio.run(run_all())
    print("\nALL ROUND-DEADLINE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
