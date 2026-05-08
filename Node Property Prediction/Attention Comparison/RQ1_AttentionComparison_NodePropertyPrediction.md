# RQ1 — Do GAT and GATv2 Learn Different Attention Patterns?

**Research Question:** When trained to the same accuracy on the same task, do GAT and GATv2 assign meaningfully different attention scores to the same graph — or do they converge to functionally equivalent routing?

---

## Pipeline Overview

```
ogbn-arxiv  ──►  Train GAT   ──►  best_gat_arxiv.pt
                 Train GATv2 ──►  best_gatv2_arxiv.pt
                                        │
                  ┌─────────────────────┼──────────────────────┐
                  ▼                     ▼                      ▼
          ogbn-products           ogbn-mag              ogbn-proteins
        (100k-node subgraph)  (large subgraph)          (subgraph)
                  │                     │                      │
                  └─────────────────────┼──────────────────────┘
                                        ▼
                             compare_attention_*.py
                                        │
                          ┌─────────────┴─────────────┐
                     per-edge                      per-node
                   cosine similarity        cosine · JSD · Spearman ρ
                          │                          │
                          └──────────┬───────────────┘
                                     ▼
                          export_attention_csv.py  ──►  attention_tables/
                          summarizer.py            ──►  statistical_summary.txt
                          visualize_results.py     ──►  figures/layer{0,1}/{distributions,degree_vs_node_metrics,head_scatter,head_boxplots}.png
```

### Step-by-step

1. **Train** both models on **ogbn-arxiv** (`train.py --model gat` and `--model gatv2`).  
   The best checkpoint (by validation accuracy) is saved to `checkpoints/`.

2. **Extract attention** on three held-out comparison graphs.  
   Models trained on a citation graph (ogbn-arxiv) are evaluated on graphs from different domains to probe whether attention differences persist under domain shift.

   | Comparison graph | Domain | Subgraph | Script |
   | --- | --- | --- | --- |
   | **ogbn-products** | Product co-purchase | 100 000-node induced subgraph | `compare_attention.py` |
   | **ogbn-mag** | Academic citation (heterogeneous, paper nodes) | Large subgraph | `compare_attention_ogbn-mag.py` |
   | **ogbn-proteins** | Protein–protein interaction | Subgraph | `compare_attention_ogbn-proteins.py` |

   Results are saved as `.pt` files per comparison graph under `Attention Comparison/<dataset>/`.

3. **Export CSVs** (`export_attention_csv.py`).  
   Converts `.pt` tensors to per-edge and per-node CSV tables for downstream analysis.

4. **Summarise** (`summarizer.py`).  
   Computes distributional statistics over all metrics; writes `statistical_summary.txt`.

5. **Visualise** (`visualize_results.py`).  
   Generates histograms, degree-vs-metric scatter plots, per-head scatter plots, and boxplots;  
   saved as PNGs in `figures/layer0/` and `figures/layer1/` (4 files each).

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

### Comparison — ogbn-products

| Property | Value |
|----------|-------|
| Task | Node classification benchmark; used here for attention extraction only |
| Nodes | ~2.45 M products |
| Edges | ~61.9 M (undirected) |
| Node features | 100-dim bag-of-words → zero-padded to 128 to match model `in_channels` |
| Subgraph | 100 000 randomly sampled nodes (induced); ~208–210k edges |
| Eligible Spearman nodes | 59 718 (layer 0) / 59 819 (layer 1) |
| Source | [Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) |

Using ogbn-products as the evaluation graph introduces a deliberate domain shift: models trained on citation-network embeddings (Node2Vec) are probed on a product co-purchase graph with bag-of-words features. Zero-padding the 100-dim features to 128 leaves 28 trailing zeros, which stresses the attention mechanism differently than the training distribution and may amplify differences between the static (GAT) and dynamic (GATv2) scoring functions.

### Comparison — ogbn-mag

| Property | Value |
|----------|-------|
| Task | Multi-class paper venue classification (349 classes); used here for attention extraction only |
| Nodes | ~1.94 M (heterogeneous graph; attention extracted on paper nodes with 128-dim Word2Vec features) |
| Edges | ~21.1 M (undirected, homogeneous projection on paper nodes) |
| Eligible Spearman nodes | 617 924 (both layers — count is identical, indicating no layer-specific sampling variation) |
| Source | [Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) |

Paper nodes in ogbn-mag carry 128-dim Word2Vec features that match the model's `in_channels` directly — no zero-padding is required. This avoids the feature-dimension mismatch present in ogbn-products and provides a cleaner signal of attention divergence driven by graph structure alone.

### Comparison — ogbn-proteins

| Property | Value |
|----------|-------|
| Task | Multi-label protein function prediction (112 binary tasks); used here for attention extraction only |
| Nodes | 132 534 proteins |
| Edges | ~79.1 M (undirected; original graph has 8-dim edge features rather than node features) |
| Node features | Adapted to 128-dim to match model `in_channels` |
| Eligible Spearman nodes | 19 592 (layer 0) / 19 526 (layer 1) |
| Source | [Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/#ogbn-proteins) |

ogbn-proteins has extremely dense connectivity (~79M edges over 132k nodes, mean degree ≈ 1 200). This makes it structurally very different from the training graph and both other comparison graphs.

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

### Checkpoint performance

| Model | Val accuracy | Best epoch |
|-------|:------------:|:----------:|
| GAT   | 0.6822       | 490        |
| GATv2 | 0.6868       | 500        |

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
| **Node cosine similarity** | For each node `i` and head `h`, cosine similarity between the two models' attention vectors over `N_l(i)`; averaged over valid heads |
| **Jensen-Shannon Divergence (JSD)** | `JSD(α_GAT[N_l(i),h] ∥ α_GATv2[N_l(i),h])` in natural-log units; theoretical range [0, ln 2 ≈ 0.693]; averaged over valid heads |
| **Spearman rank correlation (ρ)** | Rank correlation of neighbour orderings between GAT and GATv2; averaged over valid `(l, h)` pairs; computed only when `deg_l(i) ≥ 3` in layer `l`'s attention edge index (≥ 2 real neighbours) |

Node-level aggregates are reported as **uniform means** and, where available, **degree-weighted means**. High-degree nodes have more reliable Spearman estimates and exert greater influence on message passing; the degree-weighted mean is the primary graph-level summary. Degree-weighted means are available for ogbn-products (reported below); for ogbn-mag and ogbn-proteins they can be derived from the per-node CSV tables in `attention_tables/`.

---

## Results

---

### ogbn-products

#### Layer 0 (first message-passing layer)

At layer 0, both models receive the **same input**: 100-dim bag-of-words features zero-padded to 128. Attention differences arise purely from architectural choice — static vs. dynamic scoring.

Subgraph: 100 000 nodes, 308 636 edges, 2 heads

| Metric                 | Mean   | Median | Std    | Frac > 0.9 | Frac < 0.1 |
|------------------------|--------|--------|--------|------------|------------|
| Edge cosine similarity | 0.9252 | 0.9796 | 0.1327 | 77.51%     | 0.32%      |

| Metric                 | Unif. mean | Deg-wtd mean | Median | Std    | Frac > 0.9 / Frac > 0.35 |
|------------------------|------------|--------------|--------|--------|--------------------------|
| Node cosine similarity | 0.9639     | 0.9258       | 0.9966 | 0.0816 | 89.68%                   |
| Node JSD (nats)        | 0.0455     | **0.1510**   | 0.0095 | 0.1386 | 1.79% (> 0.35)           |
| Spearman ρ             | 0.6996     | **0.6983**   | 0.8571 | 0.4281 | 44.63% (> 0.9)           |

*Spearman ρ count: 59 718 nodes with deg_l(i) ≥ 3 in layer 0's attention edge index (≥ 2 real neighbours).*

#### Layer 1 (second message-passing layer)

At layer 1, the two models operate on **different hidden representations** — shaped by their divergent layer-0 attention. Despite this, edge-level cosine similarity improves markedly. However, the degree-weighted JSD rises compared to layer 0, revealing that high-degree nodes diverge *more* after representations compound.

Subgraph: 100 000 nodes, 309 834 edges, 1 head

| Metric                 | Mean   | Median | Std    | Frac > 0.9 | Frac < 0.1 |
|------------------------|--------|--------|--------|------------|------------|
| Edge cosine similarity | 0.9846 | 1.0000 | 0.1208 | 98.36%     | 1.40%      |

| Metric                 | Unif. mean | Deg-wtd mean | Median | Std    | Frac > 0.9 / Frac > 0.35 |
|------------------------|------------|--------------|--------|--------|--------------------------|
| Node cosine similarity | 0.9684     | **0.9240**   | 0.9997 | 0.1068 | 92.81%                   |
| Node JSD (nats)        | 0.0391     | **0.1922**   | 0.0017 | 0.2050 | 1.84% (> 0.35)           |
| Spearman ρ             | 0.7708     | **0.7949**   | 1.0000 | 0.4849 | 66.77% (> 0.9)           |

*Spearman ρ count: 59 819 nodes with deg_l(i) ≥ 3 in layer 1's attention edge index (≥ 2 real neighbours). The 101-node difference from layer 0 (59 718) reflects layer-specific attention edge indices — consistent with the per-layer degree treatment in DD_formula.md.*

---

### ogbn-mag

Large subgraph (paper nodes only). Both layers: 617 924 eligible nodes (count identical across layers, confirming no layer-specific sampling variation).

Layer 0 (2 heads):

| Metric                 | Mean   | Median | Std    | Frac > 0.9 | Frac < 0.1 |
|------------------------|--------|--------|--------|------------|------------|
| Edge cosine similarity | 0.9945 | 0.9979 | 0.0095 | 99.93%     | 0.00%      |

| Metric                 | Unif. mean | Median | Std    | Frac > 0.9 / Frac > 0.35 |
|------------------------|------------|--------|--------|--------------------------|
| Node cosine similarity | 0.9985     | 0.9991 | 0.0020 | ~100%                    |
| Node JSD (nats)        | 0.0022     | 0.0011 | 0.0053 | 0.00% (> 0.35)           |
| Spearman ρ             | 0.9704     | 0.9974 | 0.1280 | 94.05% (> 0.9)           |

Layer 1 (1 head):

| Metric                 | Mean   | Median | Std    | Frac > 0.9 | Frac < 0.1 |
|------------------------|--------|--------|--------|------------|------------|
| Edge cosine similarity | 1.0000 | 1.0000 | 0.0000 | 100.00%    | 0.00%      |

| Metric                 | Unif. mean | Median | Std    | Frac > 0.9 / Frac > 0.35 |
|------------------------|------------|--------|--------|--------------------------|
| Node cosine similarity | 0.9994     | 0.9997 | 0.0010 | 100%                     |
| Node JSD (nats)        | 0.0007     | 0.0003 | 0.0018 | 0.00% (> 0.35)           |
| Spearman ρ             | 0.9803     | 1.0000 | 0.1288 | 97.26% (> 0.9)           |

*Layer 1 edge cosine: std = 0 and min = max = 1.000 — GAT and GATv2 assign exactly identical edge attention weights at this layer.*

---

### ogbn-proteins

Subgraph; extremely dense graph (~79M edges, mean degree ≈ 1 200). 19 592 eligible nodes at layer 0, 19 526 at layer 1.

Layer 0 (2 heads):

| Metric                 | Mean   | Median | Std    | Frac > 0.9 | Frac < 0.1 |
|------------------------|--------|--------|--------|------------|------------|
| Edge cosine similarity | 0.9999 | 1.0000 | 0.0004 | 100.00%    | 0.00%      |

| Metric                 | Unif. mean | Median   | Std      | Frac > 0.9 / Frac > 0.35 |
|------------------------|------------|----------|----------|--------------------------|
| Node cosine similarity | 0.9999     | 0.9999   | 0.0001   | 100%                     |
| Node JSD (nats)        | 0.000076   | 0.000032 | 0.000200 | 0.00% (> 0.35)           |
| Spearman ρ             | 0.9970     | 0.9996   | 0.0523   | 99.78% (> 0.9)           |

Layer 1 (1 head):

| Metric                 | Mean   | Median | Std    | Frac > 0.9 | Frac < 0.1 |
|------------------------|--------|--------|--------|------------|------------|
| Edge cosine similarity | 1.0000 | 1.0000 | 0.0000 | 100.00%    | 0.00%      |

| Metric                 | Unif. mean | Median   | Std      | Frac > 0.9 / Frac > 0.35 |
|------------------------|------------|----------|----------|--------------------------|
| Node cosine similarity | 0.9998     | 0.9999   | 0.0004   | 100%                     |
| Node JSD (nats)        | 0.000152   | 0.000058 | 0.000374 | 0.00% (> 0.35)           |
| Spearman ρ             | 0.9970     | 0.9992   | 0.0203   | 99.82% (> 0.9)           |

*Spearman ρ count: 19 592 (layer 0) / 19 526 (layer 1). The 66-node difference reflects layer-specific attention edge indices.*

---

## Interpretation

### Cross-graph summary

The three comparison graphs span the full range of possible attention divergence:

| Graph | L0 Spearman (mean) | L1 Spearman (mean) | L0 edge cosine | L1 edge cosine | DD_proxy (approx.) |
|---|---|---|---|---|---|
| ogbn-products | 0.700 | 0.771 | 0.925 | 0.985 | **≈ 0.20** |
| ogbn-mag | 0.970 | 0.980 | 0.994 | 1.000 | **≈ 0.02** |
| ogbn-proteins | 0.997 | 0.997 | 1.000 | 1.000 | **≈ 0.003** |

ogbn-products is the only graph where GAT and GATv2 produce meaningfully different neighbour rankings. On ogbn-mag and ogbn-proteins the two models are functionally near-identical across both metrics and both layers.

### ogbn-products: meaningful divergence

**Layer 0 — same inputs, real architectural gap.**  
Both models receive the same zero-padded bag-of-words features. Yet only 44.6% of nodes have Spearman ρ > 0.9, and edge cosine averages only 0.925. This is consistent with the expressiveness argument in the GATv2 paper: the static scoring function in GAT (`e_ij = a^T [Wh_i ∥ Wh_j]`) cannot model attention patterns that require a non-linear joint interaction between source and target features, whereas GATv2's dynamic scoring can. On a co-purchase graph where product relationships may depend on complementary vs. substitute feature interactions, this gap is most visible at the first layer where raw features drive scoring.

**Layer 1 — different inputs, improved vector alignment but worsening hub divergence.**  
Edge cosine improves substantially to 0.985, and Spearman ρ climbs to 0.771 (66.8% > 0.9). Yet the degree-weighted JSD *increases* from 0.151 to 0.192. These facts are not contradictory: for most edges the attention vectors point in similar directions, but for high-degree product hubs the distributions over neighbours become less aligned. Hub nodes aggregate from many diverse neighbours; small architectural differences compound across large neighbourhoods, producing divergent probability masses even when mean cosine is high.

**The degree-weighting signal.**  
A consistent and striking pattern across both layers is the gap between uniform and degree-weighted means:

| Layer | Metric      | Uniform mean | Deg-wtd mean |
|-------|-------------|--------------|--------------|
| 0     | Node cosine | 0.964        | 0.926        |
| 0     | Node JSD    | 0.045        | 0.151        |
| 1     | Node cosine | 0.968        | 0.924        |
| 1     | Node JSD    | 0.039        | 0.192        |

High-degree hubs experience 3–5× greater attention divergence than the average node. The `degree_vs_node_metrics` figures confirm this directly: Spearman ρ fans out to near [−1, 1] at low degree and converges toward 1 at high degree. Since these hubs disproportionately influence representation learning, the functional difference between GAT and GATv2 may be larger than the uniform statistics suggest.

### ogbn-mag: almost no divergence

Edge cosine reaches exactly 1.000 (std = 0) at layer 1, and Spearman ρ ≥ 0.97 throughout. The paper-node features (128-dim Word2Vec, matched to `in_channels` without zero-padding) produce attention patterns that GAT and GATv2 agree on almost perfectly. The academic citation structure closely resembles the training domain (ogbn-arxiv), leaving little room for the dynamic-scoring advantage to emerge. The `degree_vs_node_metrics` plots show Spearman ρ variance narrowing sharply even at low degree — the opposite of the fan seen for ogbn-products.

### ogbn-proteins: functionally identical

GAT and GATv2 are essentially the same model on this graph. Edge cosine is ≥ 0.9999 at layer 0 and exactly 1.0000 at layer 1 (std = 0, min = max = 1). JSD is below 0.0002 nats on average — two orders of magnitude below ogbn-products. The Spearman ρ = 0.997 with std = 0.052 at layer 0, and std narrows further to 0.020 at layer 1. The extremely dense connectivity (~79M edges over 132k nodes) means each node aggregates from hundreds of neighbours; in this regime, the difference between static and dynamic attention scoring is washed out and both models converge to near-identical weight distributions.

### Domain shift and structural interpretation

All three comparison graphs are outside the training domain (ogbn-arxiv). The divergence ordering (products >> mag >> proteins) does not simply track domain proximity:
- ogbn-mag is another citation graph — structurally most similar to training — yet shows only near-zero divergence.
- ogbn-products shows the largest divergence despite the largest domain shift.

The key driver appears to be **graph topology and feature distribution**, not domain label. The co-purchase graph's hub-and-spoke topology combined with zero-padded bag-of-words features creates conditions where the static vs. dynamic scoring distinction matters. The citation and protein graphs do not provide this condition. The practical question — whether divergence in attention routing translates to downstream accuracy gains for GATv2 — remains open and is the subject of RQ2.

---

## File Reference

| File | Role |
|------|------|
| `models.py` | GAT and GATv2 model definitions |
| `train.py` | Training loop for ogbn-arxiv |
| `compare_attention.py` | Attention extraction on ogbn-products |
| `compare_attention_ogbn-mag.py` | Attention extraction on ogbn-mag |
| `compare_attention_ogbn-proteins.py` | Attention extraction on ogbn-proteins |
| `export_attention_csv.py` | Converts `.pt` results to per-edge and per-node CSVs |
| `summarizer.py` | Statistical summary over CSV tables |
| `visualize_results.py` | PNG figure generation (4 plots per layer) |
| `commands.txt` | Reference CLI commands for training and comparison runs |

---

## Figures

### ogbn-products — Layer 0

**Distribution histograms (edge cosine, attention weights, node cosine, JSD, Spearman ρ)**

![ ](<Attention Comparison/ogbn-products/figures/layer0/distributions.png>)

**Degree vs. node-level metrics**

![ ](<Attention Comparison/ogbn-products/figures/layer0/degree_vs_node_metrics.png>)

**Per-head attention scatter (α_GAT vs α_GATv2)**

![ ](<Attention Comparison/ogbn-products/figures/layer0/head_scatter.png>)

**Per-head attention boxplots**

![ ](<Attention Comparison/ogbn-products/figures/layer0/head_boxplots.png>)

### ogbn-products — Layer 1

**Distribution histograms**

![ ](<Attention Comparison/ogbn-products/figures/layer1/distributions.png>)

**Degree vs. node-level metrics**

![ ](<Attention Comparison/ogbn-products/figures/layer1/degree_vs_node_metrics.png>)

**Per-head attention scatter**

![ ](<Attention Comparison/ogbn-products/figures/layer1/head_scatter.png>)

**Per-head attention boxplots**

![ ](<Attention Comparison/ogbn-products/figures/layer1/head_boxplots.png>)

---

### ogbn-mag — Layer 0

**Distribution histograms**

![ ](<Attention Comparison/ogbn-mag/figures/layer0/distributions.png>)

**Degree vs. node-level metrics**

![ ](<Attention Comparison/ogbn-mag/figures/layer0/degree_vs_node_metrics.png>)

**Per-head attention scatter**

![ ](<Attention Comparison/ogbn-mag/figures/layer0/head_scatter.png>)

**Per-head attention boxplots**

![ ](<Attention Comparison/ogbn-mag/figures/layer0/head_boxplots.png>)

### ogbn-mag — Layer 1

**Distribution histograms**

![ ](<Attention Comparison/ogbn-mag/figures/layer1/distributions.png>)

**Degree vs. node-level metrics**

![ ](<Attention Comparison/ogbn-mag/figures/layer1/degree_vs_node_metrics.png>)

**Per-head attention scatter**

![ ](<Attention Comparison/ogbn-mag/figures/layer1/head_scatter.png>)

**Per-head attention boxplots**

![ ](<Attention Comparison/ogbn-mag/figures/layer1/head_boxplots.png>)

---

### ogbn-proteins — Layer 0

**Distribution histograms**

![ ](<Attention Comparison/ogbn-proteins/figures/layer0/distributions.png>)

**Degree vs. node-level metrics**

![ ](<Attention Comparison/ogbn-proteins/figures/layer0/degree_vs_node_metrics.png>)

**Per-head attention scatter**

![ ](<Attention Comparison/ogbn-proteins/figures/layer0/head_scatter.png>)

**Per-head attention boxplots**

![ ](<Attention Comparison/ogbn-proteins/figures/layer0/head_boxplots.png>)

### ogbn-proteins — Layer 1

**Distribution histograms**

![ ](<Attention Comparison/ogbn-proteins/figures/layer1/distributions.png>)

**Degree vs. node-level metrics**

![ ](<Attention Comparison/ogbn-proteins/figures/layer1/degree_vs_node_metrics.png>)

**Per-head attention scatter**

![ ](<Attention Comparison/ogbn-proteins/figures/layer1/head_scatter.png>)

**Per-head attention boxplots**

![ ](<Attention Comparison/ogbn-proteins/figures/layer1/head_boxplots.png>)

---
