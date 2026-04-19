# RQ1 — Do GAT and GATv2 Learn Different Attention Patterns?

**Research Question:** When trained to the same accuracy on the same task, do GAT and GATv2 assign meaningfully different attention scores to the same graph — or do they converge to functionally equivalent routing?

---

## Helper Notebooks

- [Training Helper Notebook (Google Colab)](https://colab.research.google.com/drive/13CBYQn_0Z5pK9tBh_L0xq5VuPWwHyzZO?usp=sharing)
- [Comparison Helper Notebook (Google Colab)](https://colab.research.google.com/drive/1jdWTIekFgjrFR7X_WryQx25vBj3yry7R?usp=sharing)

---

## Pipeline Overview

```
ogbn-arxiv  ──►  Train GAT   ──►  best_gat_arxiv.pt
                 Train GATv2 ──►  best_gatv2_arxiv.pt
                                        │
                                        ▼
              ogbl-citation2  ──►  compare_attention.py
                                   (100k-node induced subgraph)
                                        │
                          ┌─────────────┴─────────────┐
                     per-edge                      per-node
                   cosine similarity        cosine · JSD · Spearman ρ
                          │                          │
                          └──────────┬───────────────┘
                                     ▼
                          export_attention_csv.py  ──►  results/attention_tables/
                          summarizer.py            ──►  results/statistical_summary.txt
                          visualize_results.py     ──►  figures/visualisation_layer{0,1}.pdf
```

### Step-by-step

1. **Train** both models on **ogbn-arxiv** (`train.py --model gat` and `--model gatv2`).  
   The best checkpoint (by validation accuracy) is saved to `checkpoints/`.

2. **Extract attention** on **ogbl-citation2** (`compare_attention.py`).  
   Models trained on one citation graph are evaluated on a held-out citation graph to test generality.  
   A 100 000-node induced subgraph is sampled for tractability.  
   Results are saved as `.pt` files in `results/`.

3. **Export CSVs** (`export_attention_csv.py`).  
   Converts `.pt` tensors to per-edge and per-node CSV tables for downstream analysis.

4. **Summarise** (`summarizer.py`).  
   Computes distributional statistics over all metrics; writes `results/statistical_summary.txt`.

5. **Visualise** (`visualize_results.py`).  
   Generates histograms, degree-vs-metric scatter plots, per-head scatter plots, and boxplots;  
   exported as multi-page PDFs in `figures/`.

---

## Dataset

### Training — ogbn-arxiv

| Property | Value |
|----------|-------|
| Task | Node classification (40 arXiv subject categories) |
| Nodes | ~169 000 papers |
| Edges | ~1.16 M (made undirected for training) |
| Node features | 128-dim Node2Vec embeddings |
| Source | [Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) |

### Comparison — ogbl-citation2

| Property | Value |
|----------|-------|
| Task | (link prediction benchmark; used here for attention extraction only) |
| Nodes | ~2.93 M papers |
| Edges | ~30.4 M (made undirected) |
| Node features | 128-dim Node2Vec embeddings (same dimension as training → no projection needed) |
| Subgraph | 100 000 randomly sampled nodes (induced) |
| Source | [Open Graph Benchmark](https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2) |

Using ogbl-citation2 as the evaluation graph probes whether attention differences between GAT and GATv2 are specific to the training distribution or persist on an unseen graph of the same domain.

---

## Model Configuration

Both models are trained under identical hyperparameters for a fair comparison.

| Hyperparameter | Value |
|----------------|-------|
| Layers | 2 |
| Hidden channels | 64 |
| Attention heads | 2 (layer 0), 1 (layer 1 — classification head) |
| Dropout | 0.5 |
| Optimiser | Adam, lr = 0.002 |
| LR schedule | StepLR, step=100, γ=0.5 |
| Epochs | 500 |
| Gradient clipping | max-norm 1.0 |

**GAT** uses static attention: `e_ij = a^T [Wh_i ∥ Wh_j]` — the scoring vector sees source and target features independently.  
**GATv2** uses dynamic attention: `e_ij = a^T LeakyReLU(W[h_i ∥ h_j])` — the non-linearity is applied before projection, making the attention function strictly more expressive.

---

## Metrics

Attention weights are compared **layer-by-layer** across both layers of the 2-layer network.  
Layer 0 has 2 heads (intermediate); layer 1 has 1 head (classification).

### Edge-level

| Metric | Description |
|--------|-------------|
| **Edge cosine similarity** | `cos(α_GAT[e,:], α_GATv2[e,:])` across the head dimension for each edge |

### Node-level

| Metric | Description |
|--------|-------------|
| **Node cosine similarity** | For each node `i` and head `h`, cosine similarity between the two models' attention vectors over `N(i)`; averaged over heads |
| **Jensen-Shannon Divergence (JSD)** | `JSD(α_GAT[N(i),h] ∥ α_GATv2[N(i),h])` in natural-log units; range [0, ln 2 ≈ 0.693]; averaged over heads |
| **Spearman rank correlation (ρ)** | Rank correlation of neighbor orderings between GAT and GATv2; averaged over heads; undefined (NaN) for degree ≤ 1 nodes |

Node-level aggregates are reported as **degree-weighted means** — high-degree nodes have more reliable attention estimates and exert greater influence on message passing.

---

## Results

### Layer 0 (first message-passing layer)

At layer 0, both models receive the **same raw input features** (the 128-dim Node2Vec embeddings are fixed and identical for both). The difference in attention arises purely from the architectural choice of how `e_ij` is computed.

**Edge-level**

| Metric | Mean | Median | Std | Frac > 0.9 |
|--------|------|--------|-----|------------|
| Edge cosine similarity | 0.9970 | 0.9997 | 0.0077 | 99.93% |

**Node-level**

| Metric | Mean | Median | Std | Frac > 0.9 / Frac > 0.35 |
|--------|------|--------|-----|--------------------------|
| Node cosine similarity | 0.9995 | 1.0000 | 0.0015 | 99.99% |
| Node JSD | 0.0010 | 0.0000 | 0.0057 | 0.001% (> 0.35) |
| Spearman ρ | 0.7046 | 1.0000 | 0.5513 | 66.97% (> 0.9) |

### Layer 1 (second message-passing layer)

At layer 1, the two models have already diverged: the hidden representations flowing into this layer were produced by different attention mechanisms in layer 0. Despite receiving **different input features**, the two models' attention distributions converge to near-perfect agreement.

**Edge-level**

| Metric | Mean | Median | Std | Frac > 0.9 |
|--------|------|--------|-----|------------|
| Edge cosine similarity | 1.0000 | 1.0000 | 0.0000 | 100.00% |

**Node-level**

| Metric | Mean | Median | Std | Frac > 0.9 / Frac > 0.35 |
|--------|------|--------|-----|--------------------------|
| Node cosine similarity | 0.9999 | 1.0000 | 0.0005 | 100.00% |
| Node JSD | 0.0001 | 0.0000 | 0.0005 | 0.000% (> 0.35) |
| Spearman ρ | 0.7598 | 1.0000 | 0.6082 | 81.77% (> 0.9) |

---

## Interpretation

### The layer-0 / layer-1 contrast

The most striking pattern is a **reversal** between the two layers:

- **Layer 0 — same inputs, slightly diverging attention.**  
  Both models see the identical Node2Vec features. Despite this, their attention scores are not interchangeable: while the raw weight vectors are extremely close (edge cosine ≈ 0.997), the *ranking* of neighbors is less consistent — only ~67% of nodes have a Spearman ρ above 0.9. This means GAT and GATv2 sometimes prioritise different neighbors even when starting from the same features, which is exactly what the expressiveness argument in the GATv2 paper predicts: static vs. dynamic scoring can disagree on monotone rankings when key structural patterns require the interaction between source and target features.

- **Layer 1 — different inputs, near-identical attention.**  
  After layer 0, the two models have built different hidden representations. Yet their layer-1 attention distributions collapse to virtual identity (edge cosine = 1.0000, JSD ≈ 0). This suggests the intermediate representations, though geometrically distinct, induce the same *relational structure* — the transformed features encode the same neighborhood geometry, so both attention functions route information identically. The Spearman ρ also improves to 81.77%, indicating that the *orderings* converge as well.

This pattern supports the view that architectural expressiveness differences between GAT and GATv2 matter most **early in the network**, where they operate on raw features. In deeper layers, both models appear to converge on equivalent routing strategies, likely because the learned representations are shaped by the same supervised signal.

### Overall similarity

Across both layers, the JSD is essentially zero for the overwhelming majority of nodes, and node/edge cosine similarities stay above 0.999. At a coarse level, GAT and GATv2 are **functionally equivalent** attention routers on this benchmark. The practical question — whether their differences in ranking (Spearman) translate to downstream accuracy gaps — remains open and is the subject of RQ2.

---

## File Reference

| File | Role |
|------|------|
| `models.py` | GAT and GATv2 model definitions |
| `train.py` | Training loop for ogbn-arxiv |
| `compare_attention.py` | Attention extraction and metric computation |
| `export_attention_csv.py` | Converts `.pt` results to CSV |
| `summarizer.py` | Statistical summary over CSV tables |
| `visualize_results.py` | PDF figure generation |
| `commands.txt` | Reference CLI commands for training and attention comparison runs |

---

## Figures

> Embedded as base64 PNG — renders in VS Code preview and local markdown viewers.

### Layer 0 Visualisations

**Distribution histograms (edge cosine, node cosine, JSD, Spearman rho)**

![Distribution histograms](figures/layer0/distributions.png)

**Degree vs. metric scatter plots**

![Degree vs. metric scatter plots](figures/layer0/degree_vs_node_metrics.png)

**Per-head attention scatter plots**

![Per-head attention scatter plots](figures/layer0/head_scatter.png)

**Node-level metric boxplots**

![Node-level metric boxplots](figures/layer0/head_boxplots.png)

### Layer 1 Visualisations

**Distribution histograms (edge cosine, node cosine, JSD, Spearman rho)**

![Distribution histograms](figures/layer1/distributions.png)

**Degree vs. metric scatter plots**

![Degree vs. metric scatter plots](figures/layer1/degree_vs_node_metrics.png)

**Per-head attention scatter plots**

![Per-head attention scatter plots](figures/layer1/head_scatter.png)

**Node-level metric boxplots**

![Node-level metric boxplots](figures/layer1/head_boxplots.png)

---
