"""
Train GAT or GATv2 on ogbn-arxiv (node classification, 40 classes).
Saves the best checkpoint by validation accuracy.

Both models use:
    - 3 layers, 256 hidden dims, 4 attention heads
    - These match the scale used in the GATv2 paper on OGB node benchmarks
"""

import argparse
import logging
import time
from pathlib import Path
import os
import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import add_self_loops

from models import GATNodeClassifier, GATv2NodeClassifier

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train.log"),
    ],
)
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       type=str, default="gat",
                   choices=["gat", "gatv2"])
    p.add_argument("--device",      type=str, default="auto")
    p.add_argument("--epochs",      type=int, default=500)
    p.add_argument("--lr",          type=float, default=0.002)
    p.add_argument("--hidden",      type=int, default=256)
    p.add_argument("--heads",       type=int, default=4)
    p.add_argument("--num_layers",  type=int, default=3)
    p.add_argument("--dropout",     type=float, default=0.5)
    p.add_argument("--eval_steps",  type=int, default=10)
    p.add_argument("--data_dir",    type=str, default="./data")
    p.add_argument("--save_dir",    type=str, default="./checkpoints")
    return p.parse_args()


def train_epoch(model, data, train_idx, optimizer, device):
    model.train()
    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y          = data.y.squeeze(1).to(device)

    optimizer.zero_grad()
    out, _ = model(x, edge_index)
    loss = F.cross_entropy(out[train_idx], y[train_idx])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, split_idx, evaluator, device):
    model.eval()
    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)

    out, _ = model(x, edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True).cpu()

    results = {}
    for split, idx in split_idx.items():
        results[split] = evaluator.eval({
            "y_true": data.y[idx],
            "y_pred": y_pred[idx],
        })["acc"]
    return results

if __name__ == "__main__":
    args = parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device : {device}")
    log.info(f"Model  : {args.model.upper()}")

    # Dataset
    log.info("Loading ogbn-arxiv …")
    dataset   = PygNodePropPredDataset(name="ogbn-arxiv", root=args.data_dir)
    data      = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name="ogbn-arxiv")

    # ogbn-arxiv is a directed graph; make undirected for GAT (standard practice in the GATv2 paper)
    row, col = data.edge_index
    data.edge_index = torch.cat([
        torch.stack([row, col]),
        torch.stack([col, row]),
    ], dim=1)

    log.info(f"  Nodes      : {data.num_nodes:,}")
    log.info(f"  Edges      : {data.edge_index.size(1):,}  (undirected)")
    log.info(f"  Node feats : {data.x.size(1)}")
    log.info(f"  Classes    : {dataset.num_classes}")

    in_channels  = data.x.size(1)
    out_channels = dataset.num_classes

    ModelCls = GATNodeClassifier if args.model == "gat" else GATv2NodeClassifier
    model = ModelCls(
        in_channels     = in_channels,
        hidden_channels = args.hidden,
        out_channels    = out_channels,
        num_layers      = args.num_layers,
        heads           = args.heads,
        dropout         = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Parameters : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    train_idx = split_idx["train"].to(device)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path    = Path(args.save_dir) / f"best_{args.model}_arxiv.pt"
    best_val_acc = 0.0

    log.info(f"\n{'Epoch':>6}  {'Loss':>8}  {'Train':>8}  {'Val':>8}  {'Test':>8}  {'Time':>7}")
    log.info("─" * 56)

    for epoch in range(1, args.epochs + 1):
        t0   = time.time()
        loss = train_epoch(model, data, train_idx, optimizer, device)
        scheduler.step()
        elapsed = time.time() - t0

        if epoch % args.eval_steps == 0 or epoch == args.epochs:
            res = evaluate(model, data, split_idx, evaluator, device)
            marker = ""
            if res["valid"] > best_val_acc:
                best_val_acc = res["valid"]
                torch.save({
                    "epoch":     epoch,
                    "model":     args.model,
                    "state":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_acc":   res["valid"],
                    "args":      vars(args),
                }, ckpt_path)
                marker = "  ←"
            log.info(f"{epoch:>6}  {loss:>8.4f}  {res['train']:>8.4f}  "
                     f"{res['valid']:>8.4f}  {res['test']:>8.4f}  "
                     f"{elapsed:>5.1f}s{marker}")

    log.info(f"Best val acc : {best_val_acc:.4f}")
    log.info(f"Checkpoint   : {ckpt_path}")
