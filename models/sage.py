import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ModuleList
from torch_geometric.nn import SAGEConv


class SAGENodeClassifier(torch.nn.Module):
    """
    Multi-layer GraphSAGE (mean aggregation) for node classification.

    SAGEConv(in → hidden) → BN → ReLU → Dropout  [× num_layers]
    Linear(hidden → out_channels)

    No attention heads — accepts **kwargs so callers can pass heads= without error.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_ch, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin.reset_parameters()


class SAGELinkPredictor(torch.nn.Module):
    """
    Multi-layer GraphSAGE encoder + dot-product decoder for link prediction.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_ch, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

    def encode(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, z, src, dst):
        return (z[src] * z[dst]).sum(dim=-1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
