"""Network topology builders for Archipelago Gossip Consensus Simulation.

Supports fully-connected mesh and hub-and-spoke topologies.
"""

from typing import Dict, List, Union


def build_mesh(nodes: Union[int, List[str]]) -> Dict[str, List[str]]:
    """Build a fully connected mesh adjacency list.

    In a mesh network, every node has a bidirectional link to every other node.

    Args:
        nodes: Number of nodes n (generated as 'faction_0'..'faction_{n-1}'),
               or an explicit list of node identifiers.

    Returns:
        Dict mapping each node id to its list of neighbor node ids.
    """
    if isinstance(nodes, int):
        node_ids = [f"faction_{i}" for i in range(nodes)]
    else:
        node_ids = list(nodes)

    adjacency: Dict[str, List[str]] = {}
    for node_id in node_ids:
        adjacency[node_id] = [peer for peer in node_ids if peer != node_id]

    return adjacency


def build_hub_spoke(nodes: Union[int, List[str]], hub_id: str = None) -> Dict[str, List[str]]:
    """Build a hub-and-spoke adjacency list.

    One designated hub node is connected to all spokes.
    Spokes connect exclusively to the hub.

    Args:
        nodes: Number of nodes n, or explicit list of node identifiers.
        hub_id: Designated hub identifier. If None, defaults to the first node.

    Returns:
        Dict mapping each node id to its list of neighbor node ids.
    """
    if isinstance(nodes, int):
        node_ids = [f"faction_{i}" for i in range(nodes)]
    else:
        node_ids = list(nodes)

    if not node_ids:
        return {}

    hub = hub_id if hub_id is not None and hub_id in node_ids else node_ids[0]
    spokes = [nid for nid in node_ids if nid != hub]

    adjacency: Dict[str, List[str]] = {hub: list(spokes)}
    for spoke in spokes:
        adjacency[spoke] = [hub]

    return adjacency
