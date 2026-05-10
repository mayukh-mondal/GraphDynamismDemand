"""
Ego-graph subsampling for DD estimation.

Training always uses the full graph. DD is computed on G_sub(K), the union of
2-hop ego-graphs around K randomly sampled seed nodes. This preserves shared
neighborhood density by construction: every pair of neighbors of a seed node
shares that seed's neighborhood as common context.
"""

import torch
from torch_geometric.utils import k_hop_subgraph


def sample_ego_graphs(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_seeds: int,
    num_hops: int = 2,
    seed: int = 0,
):
    """
    Sample num_seeds nodes uniformly at random, then extract the union of their
    num_hops-hop ego-graphs from the full graph edge_index.

    Parameters
    ----------
    edge_index  : (2, E) full graph edges (global node indices)
    num_nodes   : total number of nodes in the full graph
    num_seeds   : K — number of seed nodes to sample
    num_hops    : ego-graph radius (default 2)
    seed        : RNG seed for reproducible seed selection

    Returns
    -------
    subset        : (N_sub,) 1-D tensor of global node indices in the subgraph
    sub_edge_index: (2, E_sub) edges re-indexed to [0, N_sub)
    mapping       : dict {local_idx: global_idx}  (same as subset[local_idx])
    """
    num_seeds = min(num_seeds, num_nodes)

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=gen)
    seed_nodes = perm[:num_seeds]

    subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=seed_nodes,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
        flow="source_to_target",
    )

    mapping = {int(local): int(global_) for local, global_ in enumerate(subset)}
    return subset, sub_edge_index, mapping
