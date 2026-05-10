import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ModuleList
from torch_geometric.nn import GATv2Conv


class GATv2NodeClassifier(torch.nn.Module):
    """
    Multi-layer GATv2 for node classification.

    Uses share_weights=True (W_l = W_r), which implements the W = [W'‖W'] constraint
    from Brody et al. (2022), matching GAT's parameter count exactly.

    Architecture mirrors GATNodeClassifier; the only difference is GATv2Conv.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels * heads
            concat = i < num_layers - 1
            n_heads = heads if concat else 1
            out_ch = hidden_channels
            self.convs.append(
                GATv2Conv(
                    in_ch,
                    out_ch,
                    heads=n_heads,
                    concat=concat,
                    dropout=dropout,
                    share_weights=True,
                )
            )
            self.bns.append(BatchNorm1d(out_ch * n_heads if concat else out_ch))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin.reset_parameters()


class GATv2LinkPredictor(torch.nn.Module):
    """
    Multi-layer GATv2 encoder + dot-product decoder for link prediction.
    Uses share_weights=True for parameter parity with GATLinkPredictor.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels * heads
            concat = i < num_layers - 1
            n_heads = heads if concat else 1
            out_ch = hidden_channels
            self.convs.append(
                GATv2Conv(
                    in_ch,
                    out_ch,
                    heads=n_heads,
                    concat=concat,
                    dropout=dropout,
                    share_weights=True,
                )
            )
            self.bns.append(BatchNorm1d(out_ch * n_heads if concat else out_ch))

    def encode(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, z, src, dst):
        return (z[src] * z[dst]).sum(dim=-1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
