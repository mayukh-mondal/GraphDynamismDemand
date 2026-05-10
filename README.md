# Dynamism Demand (DD) Pipeline — Runtime Guide

This pipeline implements the full Dynamism Demand research framework for Graph Attention Networks across four stages:

- **Stage 1** — Train GAT, GATv2, and GraphSAGE on OGB datasets.
- **Stage 2** — Extract attention weights and compute five DD metrics on ego-graph subsamples.
- **Stage 3** — Correlate DD with the GATv2 accuracy gain (RQ1), identify structural correlates of DD (RQ2), and evaluate unsupervised graph-structural proxies for DD (RQ3).

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Dependencies](#dependencies)
3. [Dataset Overview](#dataset-overview)
4. [Stage 0 — Dataset Loading](#stage-0--dataset-loading)
5. [Stage 1 — Training](#stage-1--training)
   - [Quick Start](#quick-start)
   - [All CLI Options](#all-cli-options)
   - [Per-Dataset Defaults](#per-dataset-defaults)
   - [Task Selection](#task-selection)
   - [Sampler Guide](#sampler-guide)
   - [Performance Notes](#performance-notes-graphsaint-on-large-servers)
   - [Outputs](#stage-1-outputs)
6. [Stage 2 — DD Computation](#stage-2--dd-computation)
   - [Quick Start](#quick-start-1)
   - [All CLI Options](#all-cli-options-1)
   - [DD Definitions](#dd-definitions)
   - [Subsampling](#subsampling)
   - [Memory and Performance](#memory-and-performance)
   - [Outputs](#stage-2-outputs)
7. [Stage 3 — Analysis](#stage-3--analysis)
   - [Quick Start](#quick-start-2)
   - [RQ1: DD vs AccGap](#rq1-dd-vs-accgap)
   - [RQ2: Structural Correlates](#rq2-structural-correlates)
   - [RQ3: Unsupervised Proxies](#rq3-unsupervised-proxies)
   - [Outputs](#stage-3-outputs)
8. [Running the Full Pipeline](#running-the-full-pipeline)
9. [Checkpoint Format](#checkpoint-format)
10. [Result Files](#result-files)
11. [File Reference](#file-reference)

---

## Repository Layout

```
dynamism-demand/
├── configs/
│   └── hparams.py              # Hyperparameter defaults and dataset registry
├── data/
│   └── load_dataset.py         # Stage 0: OGB dataset loading
├── models/
│   ├── __init__.py
│   ├── gat.py                  # GATConv node classifier + link predictor
│   ├── gatv2.py                # GATv2Conv (share_weights=True) — same interface
│   └── sage.py                 # SAGEConv — same interface, no attention heads
├── train/
│   └── train.py                # Stage 1: unified training script
├── extract/
│   └── extract_attention.py    # Attention weight extraction (used by Stage 2)
├── subsample/
│   └── ego_graph_sampler.py    # Ego-graph subsampling (used by Stage 2)
├── dd/
│   ├── definitions.py          # Five GPU-native DD metric functions
│   └── compute_dd.py           # Stage 2: DD computation CLI
├── analysis/
│   ├── collect_results.py      # Stage 3: aggregate Stage 2 JSONs → tidy CSVs
│   ├── structural_features.py  # Stage 3: GPU-native graph-theoretic feature computation
│   ├── rq1_dd_accgap.py        # RQ1: DD vs AccGap Spearman correlations + plots
│   ├── rq2_structural.py       # RQ2: structural feature × DD heatmap + LOO robustness
│   ├── rq3_proxies.py          # RQ3: NFD / SHS / DWCD proxy evaluation
│   └── plot_utils.py           # Shared matplotlib style, dataset colours, Theil-Sen
├── utils/
│   └── logging_utils.py        # File + console logger factory
└── scripts/
    ├── train.sh                # Example training invocation
    └── analyze.sh              # Stage 3: full analysis pipeline
```

---

## Dependencies

The pipeline uses **PyTorch** and **PyTorch Geometric** only (no other GNN frameworks).

| Package | Purpose |
|---|---|
| `torch` | Core tensors, autograd, model training |
| `torch_geometric` | GATConv, GATv2Conv, SAGEConv, NeighborLoader, GraphSAINT, k_hop_subgraph, clustering |
| `ogb` | OGB dataset download, official splits, and evaluators |
| `scipy` | `kendalltau` for RD-DD; `spearmanr`, `theilslopes` for Stage 3 analysis |
| `numpy` | Numerical aggregation throughout |
| `pandas` | Tidy DataFrames for Stage 3 aggregation and output |
| `matplotlib` | Publication-quality figures in Stage 3 |
| `scikit-learn` | `LeaveOneOut`, `LinearRegression` for RQ3 proxy evaluation |
| `tqdm` | Progress bars in Stage 2 |

All packages are standard in a PyG research environment. Install with:

```bash
pip install torch torch_geometric ogb scipy numpy pandas matplotlib scikit-learn tqdm
```

> **PyTorch 2.6+ compatibility:** PyTorch 2.6 changed `torch.load` to default to `weights_only=True`. OGB's serialised dataset cache files use pickle protocol 4 (non-tensor dicts) and require `weights_only=False`. The pipeline patches this automatically in `data/load_dataset.py` at import time and in `dd/compute_dd.py` at checkpoint load time — no user action needed.

---

## Dataset Overview

| Dataset | Task | Nodes | Metric | Sampler |
|---|---|---|---|---|
| `ogbn-arxiv` | Node classification | 169,343 | Accuracy | GraphSAINT |
| `ogbn-products` | Node classification | 2,449,029 | Accuracy | NeighborSampling |
| `ogbn-mag` | Node classification (paper subgraph) | 736,389 | Accuracy | NeighborSampling |
| `ogbn-proteins` | Node classification (multi-label) | 132,534 | ROC-AUC | NeighborSampling |
| `ogbl-collab` | Link prediction | 235,868 | Hits@50 | Full batch |
| `ogbl-citation2` | Link prediction | 2,927,963 | MRR | NeighborSampling |

**Special cases handled automatically:**

- **ogbn-proteins**: Has no node features. The 8-dimensional edge features are aggregated to nodes via `scatter_add` along `edge_index[0]`, producing an `(N, 8)` feature matrix.
- **ogbn-mag**: Heterogeneous graph. Only the `("paper", "cites", "paper")` subgraph is extracted and used.

---

## Stage 0 — Dataset Loading

Stage 0 is not a standalone script. It is a shared library module invoked automatically by both Stage 1 and Stage 2.

**File:** `data/load_dataset.py`

**API:**
```python
from data.load_dataset import load_dataset
data, split_idx, evaluator = load_dataset(name, root)
```

- `data` — `torch_geometric.data.Data` with `data.x`, `data.y`, `data.edge_index`, and `data.train_idx`, `data.val_idx`, `data.test_idx` attached as tensors.
- `split_idx` — raw OGB split dict (passed to evaluator helpers).
- `evaluator` — OGB `Evaluator` object (node or link, depending on dataset).

Datasets are downloaded automatically to `root` on first use, then cached.

---

## Stage 1 — Training

**File:** `train/train.py`

Trains one of three models on one OGB dataset for `--num_runs` independent runs (different random seeds). Saves the best-validation checkpoint per run and logs every epoch to both a log file and stdout.

### Quick Start

Using dataset defaults (recommended for reproducibility):

```bash
# Train GAT on ogbn-arxiv
python -m train.train \
    --dataset ogbn-arxiv \
    --task node_classification \
    --model gat \
    --use_default_hparams \
    --num_runs 10 \
    --data_dir .data/ogb \
    --save_dir .data/ckpts \
    --log_dir .data/logs

# Train GATv2 on the same dataset and task
python -m train.train \
    --dataset ogbn-arxiv \
    --task node_classification \
    --model gatv2 \
    --use_default_hparams \
    --num_runs 10 \
    --data_dir .data/ogb \
    --save_dir .data/ckpts \
    --log_dir .data/logs

# Train GraphSAGE
python -m train.train \
    --dataset ogbn-arxiv \
    --task node_classification \
    --model sage \
    --use_default_hparams \
    --num_runs 10 \
    --data_dir .data/ogb \
    --save_dir .data/ckpts \
    --log_dir .data/logs
```

> **Important:** For DD comparisons to be valid, GAT, GATv2, and SAGE must all be trained with the **same** `--dataset` and `--task` values.

### All CLI Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | *(required)* | OGB dataset name. One of: `ogbn-arxiv`, `ogbn-products`, `ogbn-mag`, `ogbn-proteins`, `ogbl-collab`, `ogbl-citation2`. |
| `--task` | str | dataset default | Task identifier. See [Task Selection](#task-selection). |
| `--model` | str | *(required)* | `gat`, `gatv2`, or `sage`. |
| `--num_layers` | int | from defaults | Number of GNN layers. |
| `--hidden_channels` | int | from defaults | Hidden dimension per head. |
| `--heads` | int | 4 | Number of attention heads (GAT/GATv2 only; ignored for SAGE). |
| `--dropout` | float | 0.5 | Dropout rate applied after each layer. |
| `--lr` | float | from defaults | Adam learning rate. |
| `--epochs` | int | from defaults | Training epochs per run. |
| `--num_runs` | int | 10 | Independent runs with seeds 0 … num_runs-1. |
| `--sampler` | str | from defaults | Mini-batch strategy: `neighbor`, `saint`, or `full`. |
| `--batch_size` | int | from defaults | Batch size for `neighbor`/`saint` samplers. |
| `--num_neighbors` | str | from defaults | Comma-separated fan-outs per layer, e.g. `"10,10,10"`. Required for `neighbor` sampler. |
| `--saint_walk_length` | int | 2 | Walk length for GraphSAINT random walks. |
| `--saint_num_steps` | int | 30 | Steps (batches) per epoch for GraphSAINT. |
| `--saint_sample_coverage` | int | 50 | Target node visits per node for SAINT normalisation. Higher = more accurate norm weights but longer startup. `0` disables normalisation. |
| `--saint_norm_cache_dir` | str | auto | Directory to cache SAINT normalisation statistics. Auto-set to `{data_dir}/saint_norm_cache` so they are computed once and reused across all subsequent runs on the same dataset. |
| `--saint_num_workers` | int | 2 | DataLoader worker processes for SAINT subgraph sampling. Workers run in separate processes, overlapping CPU random-walk sampling with GPU forward/backward passes. `0` = serial (no overlap). |
| `--num_cpu_threads` | int | 8 | PyTorch intra-op CPU threads per process (OMP/MKL). Without a cap, `torch_sparse` random walks spawn one thread per core, causing load averages >200 on large servers and starving the GPU. With `--saint_num_workers=2` this gives 2×8=16 cores for sampling. |
| `--data_dir` | str | *(required)* | Root directory for OGB dataset download and cache. |
| `--save_dir` | str | *(required)* | Root directory for checkpoint files. |
| `--log_dir` | str | *(required)* | Root directory for log files. |
| `--patience` | int | 50 | Early stopping patience in epochs. If validation metric does not improve by more than `--es_min_delta` for this many consecutive epochs, training stops and the best checkpoint (already saved) is kept. Set `0` to disable early stopping entirely. |
| `--es_min_delta` | float | 1e-4 | Minimum absolute increase in validation metric that resets the patience counter. Prevents tiny numerical noise from being counted as improvement. |
| `--use_default_hparams` | flag | off | Fill any `None` hyperparameters from `configs/hparams.py` dataset defaults. Explicit flags always override. |
| `--device` | str | auto | `cpu` or `cuda`. Auto-detected if not set. |

### Per-Dataset Defaults

These values are used when `--use_default_hparams` is passed and the corresponding flag is not explicitly set:

| Dataset | `--num_layers` | `--hidden_channels` | `--heads` | `--lr` | `--epochs` | `--sampler` | `--batch_size` | `--num_neighbors` |
|---|---|---|---|---|---|---|---|---|
| `ogbn-arxiv` | 3 | 256 | 4 | 0.01 | 500 | `saint` | 10000 | — |
| `ogbn-products` | 3 | 128 | 4 | 0.001 | 20 | `neighbor` | 1024 | 10,10,10 |
| `ogbn-mag` | 2 | 256 | 4 | 0.01 | 100 | `neighbor` | 1024 | 10,10 |
| `ogbn-proteins` | 6 | 64 | 4 | 0.01 | 1000 | `neighbor` | 1024 | 5,5,5,5,5,5 |
| `ogbl-collab` | 3 | 64 | 4 | 0.001 | 400 | `full` | — | — |
| `ogbl-citation2` | 3 | 256 | 4 | 0.0005 | 100 | `neighbor` | 1024 | 10,10,10 |

### Task Selection

The `--task` argument controls what the model optimises and is embedded in checkpoint paths and log filenames to prevent cross-task contamination.

| Dataset | Valid `--task` values | Notes |
|---|---|---|
| `ogbn-arxiv` | `node_classification` | Only option (default). |
| `ogbn-products` | `node_classification` | Only option (default). |
| `ogbn-mag` | `node_classification` | Only option (default). |
| `ogbn-proteins` | `all`, `0` … `111` | `all` trains jointly on all 112 protein-function labels. An integer selects a single binary subtask. |
| `ogbl-collab` | `link_prediction` | Only option (default). |
| `ogbl-citation2` | `link_prediction` | Only option (default). |

**ogbn-proteins subtask example:**
```bash
# Train on subtask 42 only (binary classification)
python -m train.train \
    --dataset ogbn-proteins --task 42 --model gatv2 \
    --use_default_hparams --num_runs 10 \
    --data_dir .data/ogb --save_dir .data/ckpts --log_dir .data/logs
```

Passing an invalid `--task` for a dataset raises a clear error listing valid choices.

### Sampler Guide

**`full`** — loads the entire graph to GPU each iteration. Only feasible for small graphs (`ogbl-collab`).

**`neighbor`** — `NeighborLoader` samples a fixed fan-out neighbourhood around each seed node. Controlled by `--num_neighbors` (one integer per layer). Used by `ogbn-products`, `ogbn-mag`, `ogbn-proteins`, `ogbl-citation2`.

**`saint`** — `GraphSAINTRandomWalkSampler` samples random-walk subgraphs. Loss is weighted by `node_norm` for an unbiased estimate. Controlled by `--saint_walk_length` and `--saint_num_steps`. Used by `ogbn-arxiv`.

Normalisation statistics (`node_norm`, `edge_norm`) are computed once by running the sampler for `--saint_sample_coverage` × N node visits, then saved to `--saint_norm_cache_dir`. On every subsequent run the cached file is loaded instantly — the startup cost is paid only once per dataset.

### Performance Notes (GraphSAINT on large servers)

GraphSAINT's random-walk subgraph sampling (`torch_sparse.random_walk` + `saint_subgraph`) is OpenMP-parallelised. On servers with 100+ cores, PyTorch's default behaviour of using all cores causes two problems:

1. **Thread-explosion overhead.** Each small batch sample spawns O(num_cores) OS threads. Thread creation and barrier synchronisation dominate the actual work, resulting in load averages >200 while GPU SM% stays around 20%.

2. **GPU starvation.** With `num_workers=0` (serial), the training loop cannot feed the GPU until the CPU finishes sampling the next batch. The GPU idles between steps.

The defaults address both:

| Setting | Value | Effect |
|---|---|---|
| `OMP_NUM_THREADS` (env) | 8 | Set in `scripts/train.sh` before any process starts — limits OpenMP at library-load time |
| `--num_cpu_threads` | 8 | Applied in `main()` via `torch.set_num_threads` — limits PyTorch's own thread pool |
| `--saint_num_workers` | 2 | Two worker processes each pre-sample the next subgraph while the GPU processes the current one — CPU and GPU run in parallel |
| `pin_memory` | auto | Enabled when `num_workers > 0` and device is CUDA — batches are placed in page-locked RAM so the H2D DMA transfer runs asynchronously |
| `non_blocking=True` | auto | Applied in the training loop `.to(device)` call — the PCIe transfer overlaps with `optimizer.zero_grad()` |
| `persistent_workers` | auto | Worker processes are kept alive between epochs — no re-fork overhead per epoch |

With 2 workers × 8 OMP threads = 16 cores for sampling running in parallel with the GPU, epoch time should be bounded by `max(T_sample_per_worker, T_gpu)` rather than `T_sample + T_gpu`.

To tune: if sampling is still the bottleneck (GPU SM% < 70%), increase `--saint_num_workers` to 4 or raise `--num_cpu_threads`. When running many jobs in parallel, lower both to reduce inter-job core contention.

### Stage 1 Outputs

**Checkpoints** — saved at:
```
{save_dir}/{dataset}/{task}/{model}_run{run:02d}_best.pt
```
Example: `.data/ckpts/ogbn-arxiv/node_classification/gat_run00_best.pt`

Each checkpoint is a dict:
```python
{
    "model_state":     model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch":           int,          # epoch of best validation metric
    "val_metric":      float,        # best validation metric
    "test_metric":     float,        # test metric at best validation epoch
    "run":             int,
    "args":            dict,         # full argparse namespace — used by Stage 2
}
```

**Logs** — written to:
```
{log_dir}/{dataset}_{task}_{model}.log
```
Each line: `YYYY-MM-DD HH:MM:SS | Run 00 | Epoch 0001 | Loss 1.2345 | Val 0.6543 | Test 0.6432 | Time 3.2s`

If early stopping fires: `Run 00 | Early stopping at epoch 312 (50 epochs without >1e-04 improvement)`

Final summary line per run: `Run 00 DONE | Best Val 0.7123 | Test@BestVal 0.7041`

Overall summary: `Summary (10 runs) | Val: 0.712 ± 0.005 | Test: 0.704 ± 0.004`

---

## Stage 2 — DD Computation

**File:** `dd/compute_dd.py`

Loads trained GAT and GATv2 checkpoints, extracts attention weights on ego-graph subsamples, and computes all five DD definitions across runs and subsample sizes. Fully decoupled from Stage 1 — does not import from `train/train.py`.

> SAGE checkpoints are not loaded here. SAGE has no attention mechanism, so it does not contribute attention weights. Its test metric is available in its own checkpoint for AccGap calculations downstream.

### Quick Start

```bash
python -m dd.compute_dd \
    --dataset ogbn-arxiv \
    --task node_classification \
    --ckpt_dir .data/ckpts \
    --data_dir .data/ogb \
    --output_dir .data/dd_results \
    --log_dir .data/logs \
    --num_runs 10 \
    --num_seeds 500 \
    --subsample_sizes "50,100,500,1000" \
    --k_list "1,3,5" \
    --layer last \
    --device cpu
```

Both `gat_run{run:02d}_best.pt` and `gatv2_run{run:02d}_best.pt` must exist under `{ckpt_dir}/{dataset}/{task}/` before running Stage 2.

### All CLI Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | *(required)* | OGB dataset name (same as Stage 1). |
| `--task` | str | dataset default | Task identifier (must match Stage 1 `--task`). |
| `--ckpt_dir` | str | *(required)* | Root checkpoint directory (`--save_dir` from Stage 1). |
| `--data_dir` | str | *(required)* | OGB dataset root (`--data_dir` from Stage 1). |
| `--output_dir` | str | *(required)* | Directory for output JSON and CSV files. |
| `--log_dir` | str | *(required)* | Directory for log files. |
| `--num_runs` | int | 10 | How many run checkpoints to process (runs 0 … num_runs-1). |
| `--num_seeds` | int | 500 | K seed nodes for the primary DD computation. Results for this K are written to the summary CSV. |
| `--subsample_sizes` | str | `"50,100,500,1000"` | Comma-separated list of K values to try. All are stored in the JSON for convergence analysis. |
| `--k_list` | str | `"1,3,5"` | k values for TKI-DD (comma-separated). Produces `tki_dd_1`, `tki_dd_3`, `tki_dd_5` columns. |
| `--layer` | str | `"last"` | Which GNN layer's attention to use. `"last"` uses the final layer (default). `"all"` computes DD at every layer. An integer selects a specific layer by index. |
| `--device` | str | auto | `cpu` or `cuda`. |
| `--num_cpu_threads` | int | 8 | PyTorch intra-op CPU thread cap (OMP/MKL). |
| `--pair_chunk` | int | 0 | Max pair-destination entries processed per GPU batch in CQ/TKI/RD-DD. Reduces peak VRAM at the cost of speed. 0 = no chunking (fastest). Typical values: 4 000 000 (≈400 MB overhead per batch). |
| `--neighbor_cap` | int | 0 | Cap the number of source nodes considered per destination node before pair generation. Prevents quadratic blowup from hub nodes (e.g. degree=5 000 → C(5000,2)=12.5 M pairs; cap=50 → 1 225). Essential for dense graphs (ogbn-proteins, ogbn-mag). 0 = no cap. |
| `--num_hops` | int | 2 | Ego-graph radius for subsampling. 2-hop gives richer shared neighbourhoods but produces very large subgraphs on dense graphs — ogbn-mag with K=500 at 2 hops produces 3.7 M edges, exhausting a 48 GB GPU. Use `--num_hops 1` for dense graphs. |

### DD Definitions

All five definitions are computed on an ego-graph subsample of the full graph using attention weights from the full-graph-trained model (not re-trained).

**Notation:**
- `α^{v2}_{ij}` — GATv2 attention weight, query node i, key node j
- `α^{gat}_{ij}` — GAT attention weight, same
- `N_i` — neighbourhood of node i in the subgraph
- `S_{ii'}` — `N_i ∩ N_{i'}`, shared neighbours of i and i'
- All pair-based definitions restrict to pairs where `|S_{ii'}| ≥ 2`

---

#### DD1 — Rank Disagreement (RD-DD)

**Uses:** GATv2 attention only

**Formula:**
```
τ(i,i') = Kendall's τ between rankings of {α^{v2}_{ij} : j ∈ S_{ii'}}
          and {α^{v2}_{i'j} : j ∈ S_{ii'}}

RD-DD = 1 − Σ |S_{ii'}|·τ(i,i') / Σ |S_{ii'}|
```
Clamped to [0, 1]. Pairs weighted by shared neighbourhood size.

**Interpretation:** 0 = identical rankings (fully static), 1 = reversed or uncorrelated rankings (maximally dynamic). Most direct operationalisation of Brody et al. Definitions 3.1/3.2.

**Implementation note:** Uses `scipy.stats.kendalltau` per pair. Pair enumeration is via common-neighbour inversion: for each node c, all pairs (i, i') where both i and i' have c as a neighbour are candidate pairs sharing c.

---

#### DD2 — Cross-Query Distribution Divergence (CQ-DD)

**Uses:** GATv2 attention only

**Formula:**
```
Renormalise α^{v2}_i and α^{v2}_{i'} over S_{ii'} → p_i, p_{i'}
m = (p_i + p_{i'}) / 2
JS(i,i') = ½·KL(p_i ∥ m) + ½·KL(p_{i'} ∥ m)

CQ-DD = mean_{valid pairs} [ JS(i,i') / log(2) ]
```
ε = 1e-10 smoothing applied before KL. Output in [0, 1].

**Interpretation:** Captures both rank and magnitude differences. Two distributions with identical rank order but different concentration still contribute a non-zero JS divergence.

---

#### DD3 — GAT-GATv2 Attention Discrepancy (GG-DD)

**Uses:** Both GATv2 and GAT attention

**Formula:**
```
For each node i in the subgraph:
    JS_i = JS(α^{v2}_i ∥ α^{gat}_i)  over N_i ∩ N_i^{gat}

GG-DD = mean_i [ JS_i / log(2) ]
```
O(|V|) — no pair enumeration needed.

**Interpretation:** Directly measures what static attention loses relative to dynamic attention per node. Closest logical predictor of AccGap: if GATv2 ≈ GAT everywhere, models are equivalent and AccGap should be near zero.

---

#### DD4 — Attention Entropy Variance (AEV-DD)

**Uses:** GATv2 attention only

**Formula:**
```
H_i     = −Σ_{j ∈ N_i} α^{v2}_{ij}·log(α^{v2}_{ij} + ε)
H^norm_i = H_i / log(|N_i|)          — normalised to [0, 1]

AEV-DD = Var_i(H^norm_i) / 0.25     — normalised by max possible variance
```
Nodes with degree < 2 are excluded.

**Interpretation:** High variance in entropy means some nodes attend selectively while others attend diffusely — a heterogeneity that static attention cannot express. Cheapest definition (O(|V|)).

---

#### DD5 — Top-k Instability (TKI-DD)

**Uses:** GATv2 attention only

**Formula:**
```
T_k(i)|_S = top-k neighbours of i in S_{ii'} by α^{v2}_{ij}
Jaccard_dissim(i,i',k) = 1 − |T_k(i)|_S ∩ T_k(i')|_S| / |T_k(i)|_S ∪ T_k(i')|_S|

TKI-DD_k = mean_{pairs with |S_{ii'}|≥k} [ Jaccard_dissim(i,i',k) ]
```
Computed for all k in `--k_list`. Primary metric is k=1 (most conservative; tied directly to the DictionaryLookup failure mode from Brody et al.).

**Interpretation:** GNN aggregation is dominated by highest-attention neighbours. TKI-DD measures whether the *effectively selected* neighbours differ across queries.

---

### Subsampling

DD is estimated on G_sub(K), the union of `--num_hops`-hop ego-graphs around K randomly sampled seed nodes, rather than the full graph. This is for computational tractability on large graphs.

**Why ego-graphs, not uniform node sampling?** Ego-graphs preserve shared neighbourhood density by construction: every pair of neighbours of a seed node v shares v's neighbourhood, ensuring `|S_{ii'}| ≥ 1` for many pairs. Uniform sampling would destroy this structure.

**Seed sampling:** `torch.randperm(num_nodes, generator=torch.Generator().manual_seed(run))[:K]` — seeded by run index for reproducibility and cross-run comparability.

**Convergence analysis:** Running with multiple values in `--subsample_sizes` (e.g. `"50,100,500,1000"`) generates a DD(K) vs K curve. The JSON output stores DD for every K. A stable estimate is reached when `|DD(K) − DD(K/2)| < 0.01`.

For full-graph computation, set K ≥ num_nodes (capped automatically).

### Memory and Performance

The pair-based metrics (CQ-DD, TKI-DD, RD-DD) enumerate all `C(d_j, 2)` pairs of source nodes sharing a destination j. On dense graphs, hub nodes with degree d = 5 000 generate up to 12.5 M pairs each, easily exhausting GPU memory.

Two complementary parameters control this:

**`--neighbor_cap N`** — caps per-destination source count to N before pair generation, bounding each node's pair contribution to C(N, 2). This is applied during the pair tensor build step and is the only way to prevent build-time OOM. Recommended: `--neighbor_cap 50` for 48 GB GPUs on ogbn-mag/proteins.

**`--pair_chunk B`** — processes the already-built pair tensor in batches of B entries, reducing peak VRAM during metric computation. Has no effect on build-time OOM. Useful when the pair tensor itself fits in memory but the metric computation peaks higher.

**`--num_hops 1`** — reduces the ego-graph radius from 2 to 1, dramatically shrinking the subgraph on dense graphs. ogbn-mag with K=500 at 2 hops produces 3.7 M edges (OOM during GATv2 forward pass on 48 GB); at 1 hop it produces ~100 K edges. Use for dense graphs.

Recommended settings for ogbn-mag / ogbn-proteins on a 48 GB GPU:
```
--num_hops 1 --neighbor_cap 100 --pair_chunk 4000000
```

### Stage 2 Outputs

**JSON** — full per-run, per-K results:
```
{output_dir}/{dataset}_{task}_dd_results.json
```

Structure:
```json
{
  "dataset": "ogbn-arxiv",
  "task": "node_classification",
  "num_runs": 10,
  "subsample_sizes": [50, 100, 500, 1000],
  "k_list": [1, 3, 5],
  "layer": "last",
  "runs": [
    {
      "run": 0,
      "acc_gap": 0.021,
      "by_K": {
        "500": {
          "rd_dd": 0.42, "cq_dd": 0.31, "gg_dd": 0.28,
          "aev_dd": 0.15, "tki_dd_1": 0.55, "tki_dd_3": 0.41, "tki_dd_5": 0.33
        }
      }
    }
  ],
  "summary": {
    "K_500": {
      "rd_dd_mean": 0.41, "rd_dd_std": 0.03,
      "cq_dd_mean": 0.30, "cq_dd_std": 0.02,
      ...
    }
  }
}
```

`acc_gap` is read directly from the saved checkpoint test metrics: `ckpt_v2["test_metric"] − ckpt_gat["test_metric"]`. No re-evaluation is performed.

**CSV** — one row per run, using the largest value in `--subsample_sizes` as the primary K:
```
{output_dir}/{dataset}_{task}_dd_summary.csv
```

Columns: `run, acc_gap, rd_dd, cq_dd, gg_dd, aev_dd, tki_dd_1, tki_dd_3, tki_dd_5`

> The primary K for the CSV is always `max(subsample_sizes)`, giving the most data-rich estimate. The full per-K results remain in the JSON.

**Log:**
```
{log_dir}/{dataset}_{task}_dd.log
```

---

## Stage 3 — Analysis

**Package:** `analysis/`

Consumes Stage 2 JSON outputs and answers the three research questions. All scripts are independently re-runnable and auto-discover available datasets by scanning the output directory — results accumulate as more datasets complete Stage 2.

> **Prerequisites:** activate the `graphtasker` conda environment before running any Stage 3 script, and run from the project root.

### Quick Start

```bash
conda activate graphtasker
bash scripts/analyze.sh
```

Or run each step individually:

```bash
export PYTHONPATH=$(pwd)   # ensure analysis/ package is importable

# Step 1 — Aggregate Stage 2 results into tidy DataFrames
python -m analysis.collect_results \
    --dd_dir .data/dd_results \
    --ckpt_dir .data/ckpts \
    --output_dir .data/analysis

# Step 2 — Compute structural graph features (cached; skips datasets already done)
python -m analysis.structural_features \
    --datasets ogbn-arxiv,ogbn-mag \
    --data_dir .data/ogb \
    --output_dir .data/analysis

# Step 3 — RQ1: DD vs AccGap
python -m analysis.rq1_dd_accgap  --analysis_dir .data/analysis

# Step 4 — RQ2: structural correlates
python -m analysis.rq2_structural  --analysis_dir .data/analysis

# Step 5 — RQ3: proxy evaluation
python -m analysis.rq3_proxies     --analysis_dir .data/analysis
```

### RQ1: DD vs AccGap

**File:** `analysis/rq1_dd_accgap.py`

Correlates each DD definition with the GATv2 accuracy gain (`acc_gap = test_metric_v2 − test_metric_gat`) across available datasets.

**Statistical approach:** Spearman ρ (non-parametric; appropriate for n=6 datasets with potential outliers). P-values are not reported because at n=6 any ρ > 0.83 trivially satisfies α=0.05. Instead:

- **Effect size:** Spearman ρ for each DD definition vs `acc_gap`, at the largest K.
- **LOO confidence interval:** for each held-out dataset, recompute ρ on the remaining n−1. Reports ρ_mean ± ρ_std over all LOO iterations.
- **Direction check:** sign of ρ. A negative correlation for any DD definition is itself a finding.
- **Run-level correlation:** within each dataset, ρ(DD_k, AccGap_k) across runs — shows co-variance across random seeds.
- **Convergence table:** minimum K at which |DD(K) − DD(K/2)| < 0.01 per dataset and definition.

**Scatter plots** use Theil-Sen slopes (robust to outliers) rather than OLS.

**Outputs:** `rq1_correlation_table.csv`, `rq1_scatter.pdf`, `rq1_convergence.pdf`, `rq1_convergence_table.csv`, `rq1_run_level.csv`

### RQ2: Structural Correlates

**File:** `analysis/rq2_structural.py`

Correlates structural graph features (computed by `structural_features.py`) with DD metrics across datasets. All features are GPU-native — NetworkX is not used.

| Feature | Complexity | Note |
|---|---|---|
| Mean / std degree | O(E) | `scatter_add` on ones |
| Degree entropy | O(V) | Shannon entropy of bincount |
| Homophily | O(E) | fraction of edges with matching labels |
| Mean clustering coeff | O(E · max_degree) | `torch_geometric.utils.clustering` |
| Degree assortativity | O(E) | Pearson r of (deg_i, deg_j) over edges |
| NFD (Neighbourhood Feature Diversity) | O(E · F) | two-pass scatter_mean + scatter variance |
| SHS (Structural Heterophily Score) | — | 1 − homophily |
| Spectral gap (λ₂ approx) | O(E · iters) | 5-step power iteration on D⁻¹/²AD⁻¹/² |

**Pre-registered directional hypotheses** (checked by sign of ρ, not p-value):
- H1: ρ(mean degree, DD) > 0
- H2: ρ(homophily, DD) < 0
- H3: ρ(clustering, DD) > 0
- H4: ρ(NFD, DD) > 0

**LOO robustness:** for each (feature, DD) pair, re-compute ρ on n−1 datasets. Features with high |ρ_mean| and low ρ_range are robust correlates.

**Outputs:** `rq2_heatmap.pdf`, `rq2_correlation.csv`, `rq2_loo_robustness.csv`, `rq2_hypothesis.csv`

### RQ3: Unsupervised Proxies

**File:** `analysis/rq3_proxies.py`

Evaluates pre-training proxies for DD — computable from graph structure and raw features alone, without any trained model.

| Proxy | Formula |
|---|---|
| P1 — NFD | E_v[Var_{j∈N_v}(h_j)] — neighbourhood feature variance |
| P2 — SHS | 1 − homophily |
| P3 — DWCD | E_v[deg(v) · (1 − C_v)] — degree-weighted clustering deficit |
| P4 — Composite | LOO-fit linear combination of min-max-normalised P1/P2/P3 |

Each proxy is evaluated on:
1. Spearman ρ(P, DD_best) — how well it approximates the best DD metric
2. Spearman ρ(P, acc_gap) — how well it predicts the target

`dd_best` is auto-selected as the DD definition with the highest |ρ| from `rq1_correlation_table.csv`, or can be specified with `--dd_best`.

If any proxy achieves ρ(P, acc_gap) ≥ 0.7, a threshold-based decision rule is printed:
```
Use GATv2 if P(G) > τ  (τ estimated from the dataset with AccGap ≈ 0)
```

**Outputs:** `rq3_proxy_comparison.csv`, `rq3_proxy_scatter.pdf`

### Stage 3 Outputs

All outputs are written to `--analysis_dir` (default `.data/analysis/`):

| File | Script | Contents |
|---|---|---|
| `df_runs.csv` | collect_results | One row per (dataset, run, K); all DD metrics + acc_gap |
| `df_summary.csv` | collect_results | One row per (dataset, K); mean ± std across runs |
| `structural_features.csv` | structural_features | One row per dataset; all structural features |
| `rq1_correlation_table.csv` | rq1_dd_accgap | Spearman ρ + LOO CIs per DD definition |
| `rq1_scatter.pdf` | rq1_dd_accgap | DD vs AccGap scatter (labeled by dataset) |
| `rq1_convergence.pdf` | rq1_dd_accgap | DD(K) vs K convergence curves |
| `rq1_convergence_table.csv` | rq1_dd_accgap | Minimum K for convergence per (dataset, DD) |
| `rq1_run_level.csv` | rq1_dd_accgap | Within-dataset ρ across runs |
| `rq2_heatmap.pdf` | rq2_structural | Feature × DD Spearman ρ heatmap |
| `rq2_correlation.csv` | rq2_structural | Full ρ table |
| `rq2_loo_robustness.csv` | rq2_structural | LOO ρ mean/range per (feature, DD) pair |
| `rq2_hypothesis.csv` | rq2_structural | H1–H4 direction check results |
| `rq3_proxy_comparison.csv` | rq3_proxies | ρ vs DD_best and vs acc_gap for each proxy |
| `rq3_proxy_scatter.pdf` | rq3_proxies | Proxy vs AccGap and vs DD_best scatter plots |

---

## Running the Full Pipeline

The minimal sequence for one dataset and one model triplet:

```bash
DATA=.data/ogb
CKPTS=.data/ckpts
LOGS=.data/logs
DD=.data/dd_results
DATASET=ogbn-arxiv
TASK=node_classification

# Stage 1: train all three models (can be run in parallel)
for MODEL in gat gatv2 sage; do
    python -m train.train \
        --dataset $DATASET --task $TASK --model $MODEL \
        --use_default_hparams --num_runs 10 \
        --data_dir $DATA --save_dir $CKPTS --log_dir $LOGS
done

# Stage 2: compute DD (requires gat + gatv2 checkpoints)
python -m dd.compute_dd \
    --dataset $DATASET --task $TASK \
    --ckpt_dir $CKPTS --data_dir $DATA \
    --output_dir $DD --log_dir $LOGS \
    --num_runs 10 --num_seeds 500 \
    --subsample_sizes "50,100,500,1000" \
    --k_list "1,3,5" --layer last
```

To run all six datasets, wrap the above in a loop over `$DATASET`.

```bash
# Stage 3: analysis (run after Stage 2 is complete for ≥2 datasets)
conda activate graphtasker
bash scripts/analyze.sh
```

---

## Checkpoint Format

Every Stage 1 checkpoint is self-contained and records the full training configuration:

```python
checkpoint = {
    "model_state":     OrderedDict,  # model.state_dict()
    "optimizer_state": dict,         # optimizer.state_dict()
    "epoch":           int,          # best val epoch (1-indexed)
    "val_metric":      float,        # best validation metric
    "test_metric":     float,        # test metric at that epoch
    "run":             int,          # run index (0-indexed)
    "args":            {             # full argparse namespace as dict
        "dataset":         str,
        "task":            str,
        "model":           str,
        "num_layers":      int,
        "hidden_channels": int,
        "heads":           int,
        "dropout":         float,
        "lr":              float,
        "epochs":          int,
        "sampler":         str,
        ...
    }
}
```

Stage 2 reads `checkpoint["args"]` to reconstruct the model architecture without any shared state with the training script.

---

## Result Files

| File | Stage | Format | Contents |
|---|---|---|---|
| `.data/ckpts/{dataset}/{task}/{model}_run{run:02d}_best.pt` | 1 | PyTorch | Model weights, optimizer state, metrics, full args |
| `.data/logs/{dataset}_{task}_{model}.log` | 1 | Text | Per-epoch training log + final summary |
| `.data/dd_results/{dataset}_{task}_dd_results.json` | 2 | JSON | All DD values per run per K, plus per-K summaries |
| `.data/dd_results/{dataset}_{task}_dd_summary.csv` | 2 | CSV | One row per run at `max(subsample_sizes)` |
| `.data/logs/{dataset}_{task}_dd.log` | 2 | Text | Per-run, per-K DD values + final summary |
| `.data/analysis/df_runs.csv` | 3 | CSV | One row per (dataset, run, K); all DD metrics |
| `.data/analysis/df_summary.csv` | 3 | CSV | One row per (dataset, K); mean ± std across runs |
| `.data/analysis/structural_features.csv` | 3 | CSV | One row per dataset; all structural features |
| `.data/analysis/rq1_*.{csv,pdf}` | 3 | CSV/PDF | RQ1 correlation table, scatter plots, convergence |
| `.data/analysis/rq2_*.{csv,pdf}` | 3 | CSV/PDF | RQ2 structural heatmap, LOO robustness, hypotheses |
| `.data/analysis/rq3_*.{csv,pdf}` | 3 | CSV/PDF | RQ3 proxy comparison and scatter plots |

---

## File Reference

### `configs/hparams.py`

Central registry. Exports:

- `DATASET_DEFAULTS` — dict of per-dataset default hyperparameters
- `AVAILABLE_TASKS` — dict of valid `--task` choices per dataset
- `DEFAULT_TASK` — default task per dataset (first entry of `AVAILABLE_TASKS`)
- `TASK_TYPE` — `"node_clf"` or `"link_pred"` per dataset
- `METRIC_KEY` — primary OGB evaluator metric key per dataset
- `NUM_CLASSES` — output dimension for node classification datasets

### `data/load_dataset.py`

- `load_dataset(name, root)` — returns `(data, split_idx, evaluator)` for any supported OGB dataset. Handles ogbn-proteins scatter and ogbn-mag subgraph extraction internally.
- Applies a `functools.partial` patch to `torch.load` at import time (`weights_only=False`) so OGB's `.pt` cache files load correctly under PyTorch 2.6+.

### `models/gat.py`

- `GATNodeClassifier(in_channels, hidden_channels, out_channels, num_layers, heads, dropout)` — multi-layer GAT with BN + ELU + Dropout, final Linear output layer.
- `GATLinkPredictor(in_channels, hidden_channels, num_layers, heads, dropout)` — GAT encoder with `encode(x, edge_index) → z` and dot-product `decode(z, src, dst)`.

### `models/gatv2.py`

- `GATv2NodeClassifier` / `GATv2LinkPredictor` — identical interface to the GAT variants. Uses `GATv2Conv(share_weights=True)`, which implements the W = [W'∥W'] constraint from Brody et al., matching GAT's parameter count exactly.

### `models/sage.py`

- `SAGENodeClassifier` / `SAGELinkPredictor` — identical interface. Uses `SAGEConv` (mean aggregation, no attention). Accepts `**kwargs` to silently absorb `heads=` when passed from a unified build function.

### `train/train.py`

Stage 1 entry point. Key internal functions:

- `build_model(args, in_channels, out_channels, task_type)` — constructs the requested model.
- `run_node_clf(...)` / `run_link_pred(...)` — full training loop for one run, with early stopping controlled by `--patience` and `--es_min_delta`.
- `eval_node_clf(...)` / `eval_link_pred(...)` — evaluation using OGB evaluators.
- `save_checkpoint(...)` — saves model + optimizer state + metadata.

### `subsample/ego_graph_sampler.py`

- `sample_ego_graphs(edge_index, num_nodes, num_seeds, num_hops, seed)` — returns `(subset, sub_edge_index, mapping)`. Uses `torch_geometric.utils.k_hop_subgraph`.

### `extract/extract_attention.py`

- `extract_attention(model, x_sub, sub_edge_index, device)` — iterates `model.convs` and `model.bns`, calling each conv with `return_attention_weights=True`. Returns a list of `(E_sub,)` head-averaged tensors (one per layer) without modifying the model source.
- `select_layer(attn_by_layer, layer_spec)` — filters the list by `"last"`, `"all"`, or integer index.

### `dd/definitions.py`

CPU reference implementations (used for validation) and GPU-native primary implementations:

- `build_attn_dict(sub_edge_index, alpha)` — builds `{query_node: {neighbour: weight}}` dict (CPU).
- `rd_dd` / `cq_dd` / `gg_dd` / `aev_dd` / `tki_dd` — CPU reference implementations.
- `aev_dd_gpu(sub_edge_index, alpha, num_nodes)` — AEV-DD via `scatter_add`; O(E), no Python loop.
- `gg_dd_gpu(sub_edge_index, alpha_v2, alpha_gat, num_nodes)` — GG-DD via vectorised JS divergence.
- `cq_tki_rd_dd_gpu(sub_edge_index, alpha, num_nodes, k_list, pair_chunk, neighbor_cap)` — CQ-DD, TKI-DD, and RD-DD in a single pass over the pair tensor. Pair generation via `repeat_interleave` + cumsum offsets (no Python loops). Accepts `pair_chunk` for chunked metric computation and `neighbor_cap` to cap per-destination pair count.

### `dd/compute_dd.py`

Stage 2 entry point. Key internal functions:

- `rebuild_model(saved_args, in_channels, task_type)` — reconstructs a model from a checkpoint's `args` dict without importing `train/train.py`.
- `compute_dd_for_subgraph(...)` — runs extraction + all five definitions for one (run, K) pair. All metrics computed GPU-natively (no `.cpu()` transfer for DD). Accepts `pair_chunk`, `neighbor_cap`, and `num_hops`.
- `_summarise(run_rows, k_list)` — computes mean ± std across runs.
- Loads checkpoints with `weights_only=False` (required for PyTorch 2.6+).

### `analysis/collect_results.py`

- Scans `--dd_dir` for all `*_dd_results.json` files (auto-discovers datasets).
- Flattens nested JSON into two tidy DataFrames: `df_runs` (one row per dataset/run/K) and `df_summary` (mean ± std per dataset/K).
- Also pulls SAGE test metrics from Stage 1 checkpoints to add a `sage_gap` column.

### `analysis/structural_features.py`

- Computes degree stats, homophily, clustering coefficient, degree assortativity, NFD, SHS, and spectral gap (power iteration approximation) for any OGB dataset.
- All operations are GPU-native (`scatter_add`, `torch_geometric.utils.clustering`) — no NetworkX.
- Results are cached to `structural_features.csv`; skips already-computed datasets on re-runs.

### `analysis/rq1_dd_accgap.py`

- Cross-dataset Spearman ρ table with LOO confidence intervals (`rq1_correlation_table.csv`).
- Scatter plots with Theil-Sen regression lines, dataset labels, and ±1 std error bars (`rq1_scatter.pdf`).
- Convergence plots DD(K) vs K (`rq1_convergence.pdf`) and minimum convergence K table.
- Run-level within-dataset ρ(DD_k, AccGap_k) across runs (`rq1_run_level.csv`).

### `analysis/rq2_structural.py`

- Merges structural features with DD summaries on `dataset`.
- Builds a feature × DD Spearman ρ heatmap (`rq2_heatmap.pdf`) with annotated cell values.
- Checks pre-registered directional hypotheses H1–H4 by sign of ρ (`rq2_hypothesis.csv`).
- LOO robustness: re-computes ρ for each held-out dataset, reports mean/range (`rq2_loo_robustness.csv`).

### `analysis/rq3_proxies.py`

- Builds proxies P1 (NFD), P2 (SHS), P3 (DWCD) from `structural_features.csv`.
- Fits P4 composite via LOO `LinearRegression` with min-max normalisation.
- Auto-selects `dd_best` from `rq1_correlation_table.csv` (highest |ρ| with acc_gap).
- Prints a threshold decision rule if any proxy achieves ρ(P, acc_gap) ≥ 0.7.

### `analysis/plot_utils.py`

Shared constants and helpers used by all analysis scripts:
- `set_style()` — global matplotlib rcParams for consistent font sizes, grid style, spine removal.
- `DATASET_ORDER`, `DATASET_COLORS`, `DATASET_LABELS`, `DD_DISPLAY` — consistent display across all plots.
- `annotate_points(ax, xs, ys, labels)` — dataset-name annotations on scatter plots.
- `theil_sen_line(ax, x, y)` — robust regression line overlay.

### `utils/logging_utils.py`

- `get_logger(log_dir, name)` — returns a `logging.Logger` with file handler (writes to `{log_dir}/{name}.log`) and stdout handler. Safe to call multiple times (guards against duplicate handlers).
