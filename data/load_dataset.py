"""
Stage 0 — OGB dataset loading with official train/val/test splits.

Returns (data, split_idx, evaluator) for all six datasets.
Special cases:
  ogbn-proteins  — no node features; aggregate edge_attr (8-dim) to nodes via scatter_add.
  ogbn-mag       — heterogeneous graph; extract the paper-cites-paper homogeneous subgraph.
  ogbl-*         — link prediction; split_idx contains positive/negative edge sets.
"""

import functools

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import NodePropPredDataset, Evaluator as NodeEvaluator
from ogb.linkproppred import LinkPropPredDataset, Evaluator as LinkEvaluator

# PyTorch 2.6 changed the torch.load default from weights_only=False to True.
# OGB serialised its .pt cache files with pickle protocol 4 (non-tensor dicts),
# so they require weights_only=False. Patch the default here so all torch.load
# calls originating from OGB at dataset-instantiation time see the right default,
# without modifying installed third-party files.
torch.load = functools.partial(torch.load, weights_only=False)


def load_dataset(name: str, root: str):
    """
    Load an OGB dataset.

    Returns
    -------
    data       : torch_geometric.data.Data
                 Node features in data.x, labels in data.y.
                 Splits stored as data.train_idx, data.val_idx, data.test_idx.
    split_idx  : dict  (raw OGB split dict, passed to evaluator helpers)
    evaluator  : ogb Evaluator object
    """
    if name.startswith("ogbn"):
        return _load_node_dataset(name, root)
    if name.startswith("ogbl"):
        return _load_link_dataset(name, root)
    raise ValueError(f"Unknown dataset '{name}'. Expected ogbn-* or ogbl-*.")


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

def _load_node_dataset(name: str, root: str):
    dataset = NodePropPredDataset(name=name, root=root)
    split_idx = dataset.get_idx_split()
    evaluator = NodeEvaluator(name=name)

    if name == "ogbn-mag":
        return _load_ogbn_mag(dataset, split_idx, evaluator)

    graph, label = dataset[0]

    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)

    # ogbn-arxiv is a directed citation graph. Symmetrizing (adding reverse edges)
    # is standard practice for GNN baselines and recovers ~13 points vs. directed.
    if name == "ogbn-arxiv":
        edge_index = to_undirected(edge_index, num_nodes=graph["num_nodes"])

    if name == "ogbn-proteins":
        # Proteins has no node features; aggregate 8-dim edge features to nodes.
        num_nodes = graph["num_nodes"]
        edge_attr = torch.tensor(graph["edge_feat"], dtype=torch.float32)
        x = torch.zeros(num_nodes, edge_attr.size(1), dtype=torch.float32)
        x.scatter_add_(
            0,
            edge_index[0].unsqueeze(1).expand_as(edge_attr),
            edge_attr,
        )
        y = torch.tensor(label, dtype=torch.float32)          # (N, 112) float for BCELoss
    else:
        x = torch.tensor(graph["node_feat"], dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long).squeeze(-1) # (N,) long for CrossEntropy

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=graph["num_nodes"])
    _attach_splits(data, split_idx, keys=("train", "valid", "test"))
    return data, split_idx, evaluator


def _load_ogbn_mag(dataset, split_idx, evaluator):
    """Extract the paper→cites→paper homogeneous subgraph from ogbn-mag."""
    graph, label = dataset[0]

    edge_index = torch.tensor(
        graph["edge_index_dict"][("paper", "cites", "paper")], dtype=torch.long
    )
    # paper-cites-paper is directed; symmetrize to match standard GNN baselines.
    edge_index = to_undirected(edge_index, num_nodes=graph["num_nodes_dict"]["paper"])
    x = torch.tensor(graph["node_feat_dict"]["paper"], dtype=torch.float32)
    y = torch.tensor(label["paper"], dtype=torch.long).squeeze(-1)
    num_nodes = graph["num_nodes_dict"]["paper"]

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    # split_idx for mag is nested: split_idx["train"]["paper"] etc.
    flat = {
        "train": split_idx["train"]["paper"],
        "valid": split_idx["valid"]["paper"],
        "test":  split_idx["test"]["paper"],
    }
    _attach_splits(data, flat, keys=("train", "valid", "test"))
    return data, split_idx, evaluator


def _attach_splits(data, split_idx, keys):
    for key in keys:
        attr = "val_idx" if key == "valid" else f"{key}_idx"
        setattr(data, attr, torch.tensor(split_idx[key], dtype=torch.long))


# ---------------------------------------------------------------------------
# Link prediction
# ---------------------------------------------------------------------------

def _load_link_dataset(name: str, root: str):
    dataset = LinkPropPredDataset(name=name, root=root)
    split_idx = dataset.get_edge_split()
    evaluator = LinkEvaluator(name=name)

    graph = dataset[0]

    x = (
        torch.tensor(graph["node_feat"], dtype=torch.float32)
        if graph.get("node_feat") is not None
        else None
    )
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, num_nodes=graph["num_nodes"])

    if graph.get("edge_feat") is not None:
        data.edge_attr = torch.tensor(graph["edge_feat"], dtype=torch.float32)

    return data, split_idx, evaluator
