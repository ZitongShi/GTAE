from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import subgraph, to_networkx

from .data import TextAttributedGraph


def _normalize_groups(groups: list[list[int]], clients: int) -> list[list[int]]:
    groups = [sorted(set(group)) for group in groups if group]
    while len(groups) > clients:
        groups.sort(key=len)
        first = groups.pop(0)
        groups[0].extend(first)
    while len(groups) < clients:
        groups.sort(key=len, reverse=True)
        largest = groups.pop(0)
        middle = max(1, len(largest) // 2)
        groups.extend([largest[:middle], largest[middle:]])
    return groups


def louvain_partition(graph: TextAttributedGraph, clients: int, seed: int) -> list[list[int]]:
    data = type("Graph", (), {"edge_index": graph.edge_index, "num_nodes": graph.num_nodes})()
    nx_graph = to_networkx(data, to_undirected=True)
    communities = nx.community.louvain_communities(nx_graph, seed=seed)
    return _normalize_groups([list(group) for group in communities], clients)


def dirichlet_partition(graph: TextAttributedGraph, clients: int, alpha: float, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    groups = [[] for _ in range(clients)]
    labels = graph.y.cpu().numpy()
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(clients, alpha))
        cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        for client_id, values in enumerate(np.split(indices, cuts)):
            groups[client_id].extend(values.tolist())
    return groups


def induced_client_graph(graph: TextAttributedGraph, node_ids: list[int]) -> TextAttributedGraph:
    nodes = torch.tensor(sorted(node_ids), dtype=torch.long, device=graph.edge_index.device)
    edge_index, _ = subgraph(nodes, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes)
    global_to_local = {int(global_id): local_id for local_id, global_id in enumerate(nodes.tolist())}
    texts = [graph.texts[int(index)] for index in nodes.tolist()]
    values = {
        "x": graph.x[nodes],
        "edge_index": edge_index,
        "y": graph.y[nodes],
        "texts": texts,
        "train_mask": graph.train_mask[nodes],
        "val_mask": graph.val_mask[nodes],
        "test_mask": graph.test_mask[nodes],
        "global_node_ids": nodes,
    }
    if not values["train_mask"].any():
        first = global_to_local[int(nodes[0])]
        values["train_mask"][first] = True
    return replace(graph, **values)


def partition_graph(
    graph: TextAttributedGraph,
    clients: int,
    method: str,
    seed: int,
    dirichlet_alpha: float = 0.5,
) -> list[TextAttributedGraph]:
    if method.lower() == "louvain":
        groups = louvain_partition(graph, clients, seed)
    elif method.lower() == "dirichlet":
        groups = dirichlet_partition(graph, clients, dirichlet_alpha, seed)
    else:
        order = torch.randperm(graph.num_nodes, generator=torch.Generator().manual_seed(seed))
        groups = [chunk.tolist() for chunk in torch.tensor_split(order, clients)]
    return [induced_client_graph(graph, group) for group in groups]

