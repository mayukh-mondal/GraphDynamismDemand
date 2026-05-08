"""
Shared node-classification encoders for GAT and GATv2.

Both models:
  - Accept 128-dim node features (ogbn-arxiv / ogbl-citation2)
  - Return (logits, attention_weights) where attention_weights is a list of (edge_index, alpha) tuples — one per layer
  - Use the same hyperparameter interface so they can be compared fairly
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATConv, GATv2Conv


class GATNodeClassifier(torch.nn.Module): # returns (out, attn_list) where attn_list[l] = (edge_index, alpha_l) and alpha_l has shape [E, heads]

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()

        for i in range(num_layers):
            in_ch  = in_channels        if i == 0 else hidden_channels * heads
            out_ch = out_channels       if i == num_layers - 1 else hidden_channels
            n_head = 1                  if i == num_layers - 1 else heads
            concat = (i < num_layers - 1)

            self.convs.append(
                GATConv(in_ch, out_ch, heads=n_head,
                        dropout=dropout, concat=concat,
                        add_self_loops=True)
            )
            if concat:
                self.bns.append(BatchNorm1d(out_ch * n_head))
            else:
                self.bns.append(BatchNorm1d(out_ch))

    def forward(self, x: Tensor, edge_index: Tensor):
        attn_list = []

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
            attn_list.append((ei, alpha))          # alpha: [E, heads]
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x, attn_list


class GATv2NodeClassifier(torch.nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()

        for i in range(num_layers):
            in_ch  = in_channels        if i == 0 else hidden_channels * heads
            out_ch = out_channels       if i == num_layers - 1 else hidden_channels
            n_head = 1                  if i == num_layers - 1 else heads
            concat = (i < num_layers - 1)

            self.convs.append(
                GATv2Conv(in_ch, out_ch, heads=n_head,
                          dropout=dropout, concat=concat,
                          add_self_loops=True, share_weights=False)
            )
            if concat:
                self.bns.append(BatchNorm1d(out_ch * n_head))
            else:
                self.bns.append(BatchNorm1d(out_ch))

    def forward(self, x: Tensor, edge_index: Tensor):
        attn_list = []

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
            attn_list.append((ei, alpha))
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x, attn_list
