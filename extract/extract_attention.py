"""
Attention weight extraction for GAT and GATv2 models.

Runs a forward pass on a (sub)graph without modifying the model files by
directly iterating model.convs and model.bns and calling each GATConv /
GATv2Conv with return_attention_weights=True.

SAGEConv has no attention mechanism — do not call this on SAGE models.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_attention(model, x_sub, sub_edge_index, device):
    """
    Extract per-layer, head-averaged attention weights from a GAT or GATv2
    model on a subgraph, without modifying the model source.

    Works for both NodeClassifier and LinkPredictor variants because both
    expose `model.convs` (list of GATConv/GATv2Conv) and `model.bns`
    (list of BatchNorm1d) in the same order.

    Parameters
    ----------
    model         : GATNodeClassifier | GATLinkPredictor | GATv2NodeClassifier
                    | GATv2LinkPredictor  (must have .convs and .bns)
    x_sub         : (N_sub, F) feature tensor for subgraph nodes (CPU or GPU)
    sub_edge_index: (2, E_sub) re-indexed edge tensor (CPU or GPU)
    device        : torch.device to run inference on

    Returns
    -------
    attn_by_layer : list[torch.Tensor], length = num_layers
                    Each tensor has shape (E_sub,) — head-averaged attention
                    weight per subgraph edge.  attn_by_layer[-1] is the final
                    (closest-to-output) layer, used as the default for DD.
    """
    model.eval()
    x = x_sub.to(device)
    ei = sub_edge_index.to(device)

    attn_by_layer = []

    E_sub = ei.size(1)
    for conv, bn in zip(model.convs, model.bns):
        out, (_, alpha) = conv(x, ei, return_attention_weights=True)
        # alpha: (E_sub [+ N_sub self-loops], num_heads) — slice to original
        # edges only (GATConv appends self-loops at the end when add_self_loops=True)
        attn_by_layer.append(alpha[:E_sub].mean(dim=1))
        x = bn(out)
        x = F.elu(x)
        # dropout is a no-op in eval mode; no explicit call needed

    return attn_by_layer


def select_layer(attn_by_layer, layer_spec):
    """
    Select attention tensors according to a layer specification.

    Parameters
    ----------
    attn_by_layer : list[Tensor] from extract_attention
    layer_spec    : "last" | "all" | int

    Returns
    -------
    selected : list[Tensor]  (always a list for uniform downstream handling)
               Length 1 for "last" or int; length == num_layers for "all".
    layer_indices : list[int] corresponding layer indices
    """
    n = len(attn_by_layer)
    if layer_spec == "last":
        return [attn_by_layer[-1]], [n - 1]
    if layer_spec == "all":
        return attn_by_layer, list(range(n))
    idx = int(layer_spec)
    if not (-n <= idx < n):
        raise ValueError(f"Layer index {idx} out of range for model with {n} layers.")
    return [attn_by_layer[idx]], [idx % n]
