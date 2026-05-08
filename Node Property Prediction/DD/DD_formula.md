# Dynamism Demand — Formula Reference

---

## 1. DD_proxy(G) — Operational Definition

**Question answered:** Does using GATv2 over GAT change attention routing on this graph?
This is the primary target variable for RQ2 and RQ3.

### Step 1 — Per-node, per-layer, per-head Spearman correlation

For each node $i$, layer $l$, and head $h$, define $\deg_l(i)$ as the degree of node $i$ in layer $l$'s attention edge index (which already includes the self-loop added by GATConv). Compute $\rho_i^{(l,h)}$ only when $\deg_l(i) \geq 3$:

$$\rho_i^{(l,h)} = \begin{cases} \text{Spearman}\!\left(\alpha_{\text{GAT}}^{(l,h)}[N_l(i)],\; \alpha_{\text{GATv2}}^{(l,h)}[N_l(i)]\right) & \text{if } \deg_l(i) \geq 3 \\ \text{undefined} & \text{otherwise} \end{cases}$$

where $N_l(i)$ is the set of neighbours of node $i$ in layer $l$'s attention edge index, and $\alpha^{(l,h)}[N_l(i)]$ is the vector of attention weights node $i$ assigns to its neighbours at layer $l$, head $h$.

The per-layer degree index $\deg_l(i)$ is used (rather than a single fixed $\deg(i)$) because the attention edge index can differ across layers — for example, due to mini-batch neighbourhood sampling. Nodes with $\deg_l(i) \leq 2$ yield degenerate rank correlations over only two items (rank correlations over two items can only return $\pm 1$). $\deg_l(i) = 2$ means one real neighbour plus the self-loop. We retain pairs $(i, l)$ with $\deg_l(i) \geq 3$, giving at least three ranked positions. With three ranked positions, Spearman takes four distinct values $\{-1,\,-0.5,\,0.5,\,1\}$, providing graded rather than degenerate estimates.

### Step 2 — Head-weighted aggregation across layers

Let $\mathcal{V}_i = \{(l,h) : \rho_i^{(l,h)} \text{ is defined}\}$ be the set of valid (layer, head) pairs for node $i$.

$$\bar{\rho}_i = \frac{\displaystyle\sum_{(l,h)\,\in\,\mathcal{V}_i} \rho_i^{(l,h)}}{|\mathcal{V}_i|}$$

Nodes for which $\mathcal{V}_i = \emptyset$ (no valid Spearman at any layer or head) are excluded from all subsequent steps. In practice, such exclusions are exceedingly rare for moderate batch sizes, so the degree-weighted mean is dominated by nodes with reliable estimates.

This formula gives each individual attention head equal weight over the valid pairs. A layer with more heads naturally contributes more to $\bar\rho_i$ when all its heads are valid — not as a deliberate privileging of that layer but as a consequence of head count. For the default 2-layer model ($H_0 = 2$, $H_1 = 1$) when all heads are valid ($|\mathcal{V}_i| = 3$):

$$\bar{\rho}_i = \frac{\rho_i^{(0,1)} + \rho_i^{(0,2)} + \rho_i^{(1,1)}}{3}$$

If a node has $\deg_l(i) < 3$ at exactly one layer (e.g., sampled with too few neighbours in that layer's mini-batch), the average is taken over the remaining $|\mathcal{V}_i| = 2$ or $|\mathcal{V}_i| = 1$ valid pairs, without inflating the denominator with undefined terms.

### Step 3 — Per-node dynamism demand

$$\text{dd}(i) = 1 - \max\!\left(0,\; \bar{\rho}_i\right) \;\in\; [0, 1]$$

Clamping at zero ensures anticorrelated rankings ($\bar\rho_i < 0$) do not push $\text{dd}(i)$ above 1 — anticorrelation is treated as equally demanding as zero correlation.

### Step 4 — Graph-level DD (degree-weighted mean)

$$\boxed{DD_{\text{proxy}}(G) = \frac{\displaystyle\sum_{\substack{i \in V \\ \bar{\rho}_i\,\text{defined}}} \deg(i)\cdot \text{dd}(i)}{\displaystyle\sum_{\substack{i \in V \\ \bar{\rho}_i\,\text{defined}}} \deg(i)}}$$

The eligibility condition is $\bar{\rho}_i$ defined (i.e.\ $\mathcal{V}_i \neq \emptyset$), which subsumes the $\deg_l(i) \geq 3$ requirement from Step 1. The weight $\deg(i)$ here is the full-graph degree of node $i$ (not the layer-specific sampled degree), so that hub nodes retain their true structural influence even when only a subset of their neighbours was sampled in any given layer. High-degree nodes receive greater weight because they have more reliable Spearman estimates (more ranked positions) and greater influence on message passing.

### Boundary conditions

The graph-level values below assume every eligible node exhibits the stated $\bar\rho_i$ value; in practice $DD_{\text{proxy}}$ is an average and intermediate per-node values mix.

| Scenario | $\bar\rho_i$ | $\text{dd}(i)$ | $DD_{\text{proxy}}$ |
| --- | --- | --- | --- |
| GAT $\equiv$ GATv2 everywhere | $1.0$ | $0.0$ | **0.0** — static attention suffices |
| Rankings fully uncorrelated | $0.0$ | $1.0$ | **1.0** — maximum dynamism demand |
| Rankings anticorrelated | $-1.0$ | $1.0$ (clamped) | **1.0** — treated as max demand |
| Partially correlated | $(0, 1)$ | $(0, 1)$ | **(0, 1)** — interpolated |
| Node valid at some $(l,h)$ only | average over $\mathcal{V}_i$ | $(0,1)$ | contributes with full-graph $\deg(i)$ weight |

### Proxy validity assumption

$DD_{\text{proxy}}$ is sound under the assumption that **GAT learns the best static approximation** to the true attention function. Under this assumption:

- If all query nodes agree on neighbour rankings → static attention is optimal → GAT captures it → GATv2 does not deviate → $DD_{\text{proxy}} \approx 0$
- If query nodes strongly disagree → static attention fails → GATv2 diverges from GAT → $DD_{\text{proxy}} \approx 1$

Observed statistics across three benchmarks (mean node Spearman per layer, confirmed from attention comparison data):

| Graph | Layer 0 $\bar\rho$ (mean) | Layer 1 $\bar\rho$ (mean) | Edge cosine L0 (vector similarity, not part of DD) | Edge cosine L1 (vector similarity, not part of DD) | $DD_{\text{proxy}}$ (approx.) |
| --- | --- | --- | --- | --- | --- |
| ogbn-products | 0.700 | 0.771 | 0.925 | 0.985 | **≈ 0.20** (low) |
| ogbn-mag | 0.970 | 0.980 | 0.994 | 1.000 | **≈ 0.02** (near zero) |
| ogbn-proteins | 0.997 | 0.997 | 1.000 | 1.000 | **≈ 0.003** (near zero) |

*Approximate DD values estimated from per-node Spearman means and degree statistics; exact values will be computed from the full attention extraction pipeline.*

The low $DD_{\text{proxy}}$ for ogbn-products despite non-trivial Spearman spread (~0.70) is explained by degree-weighting: the `degree_vs_node_spearman` plots show that Spearman variance fans out sharply at low degree and converges to $\rho \approx 1$ at high degree, so degree-weighted aggregation is dominated by the high-degree (high-$\rho$) hubs.

For ogbn-proteins and ogbn-mag, GAT and GATv2 are essentially identical ($\rho \approx 0.997$ and edge cosine $\approx 1.000$), yielding $DD_{\text{proxy}} \approx 0$. The low DD despite non-trivial vector divergence in ogbn-products confirms that the proxy measures routing-order agreement (Spearman rank correlation) rather than raw vector similarity — the two signals are related but distinct, and it is the ranking signal that governs which neighbours influence each node's representation. Crucially, low DD on ogbn-products is not a failure of the metric: it correctly identifies that neighbour-importance ordering is largely static on this graph, which is exactly the signal needed to determine whether GATv2's dynamic routing is warranted.

---

## 2. DD_true(G) — Structural Definition

**Question answered:** Is GATv2 genuinely computing query-dependent attention?
This matches the paper's original definition verbatim and involves GATv2 alone.

### Step 1 — Common-neighbour sets

For every ordered pair $(u, v) \in V \times V$, define:

$$C_{uv} = N(u) \cap N(v)$$

Only pairs with $|C_{uv}| \geq 2$ contribute (minimum required for Spearman).

### Step 2 — Cross-query Spearman within GATv2

$$\rho_{uv}^{(l,h)} = \text{Spearman}\!\left(\alpha_{\text{GATv2}}^{(l,h)}[u,\, C_{uv}],\;\alpha_{\text{GATv2}}^{(l,h)}[v,\, C_{uv}]\right)$$

Per-pair average across layers and heads:

$$\bar{\rho}_{uv} = \frac{\displaystyle\sum_{l}\sum_{h=1}^{H_l} \rho_{uv}^{(l,h)}}{\displaystyle\sum_l H_l}$$

### Step 3 — Graph-level DD (weighted by shared-neighbour count)

$$\boxed{DD_{\text{true}}(G) = 1 - \max\!\left(0,\;\frac{\displaystyle\sum_{\substack{(u,v):\\ |C_{uv}|\,\geq\,2}} \min(|C_{uv}|, K)\cdot \bar{\rho}_{uv}}{\displaystyle\sum_{\substack{(u,v):\\ |C_{uv}|\,\geq\,2}} \min(|C_{uv}|, K)}\right)}$$

Pairs with more shared neighbours yield more reliable Spearman estimates and receive proportionally greater weight. The cap $K$ (suggested default: $K = 50$) prevents a single high-overlap pair from dominating the weighted mean.

### Practical scope of DD_true

$DD_{\text{true}}$ is expensive to compute over all pairs. In practice, restrict to **adjacent pairs** $(u, v) \in E$ — their common neighbours are exactly the triangles containing edge $(u,v)$, which are cheap to enumerate. This covers the most structurally informative pairs and keeps complexity linear in the number of triangles.

---

## 3. Relationship Between DD_proxy and DD_true

| | $DD_{\text{proxy}}$ | $DD_{\text{true}}$ |
| --- | --- | --- |
| Involves GAT? | Yes | No |
| Requires both models? | Yes | GATv2 only |
| Measures | GAT approximation error | Query-dependence of GATv2 |
| Target for RQ2/RQ3? | **Yes** (model selection) | Validation only |
| Computational cost | $O(\lvert E\rvert)$ per layer | $O(\text{triangles})$ |

**Validation protocol (Option B):** Compute both $DD_{\text{proxy}}$ and $DD_{\text{true}}$ as graph-level scalars on each of 5 randomly drawn induced subgraphs (each ~5k nodes) from the same graph. Report the Pearson $r$ between the five $DD_{\text{true}}$ values and the five $DD_{\text{proxy}}$ values. A high correlation (target $r > 0.85$) validates that the proxy captures genuine query-dependence and not just GAT's approximation failures. This avoids the need to define a per-node version of $DD_{\text{true}}$ and makes the sanity check a single scalar comparison.

---

## 4. Interpreting DD with Accuracy

Spearman-based DD is a structural signal. Accuracy gap $\Delta\text{acc} = \text{acc}(\text{GATv2}) - \text{acc}(\text{GAT})$ is a functional signal. They answer different questions and should be used together.

### The 2×2 interpretation table

| $DD_{\text{proxy}}$ | $\Delta\text{acc}$ | Interpretation |
| --- | --- | --- |
| Low ($< 0.33$) | $\approx 0$ | Static attention suffices. GATv2 adds no value. Expected and self-consistent outcome. |
| Low ($< 0.33$) | High ($> 0$) | GATv2 gains accuracy without dynamic routing — likely better regularisation or capacity, not dynamism. *Speculative pending RQ2; if this cell is populated, it weakens the claim that $DD_{\text{proxy}}$ specifically measures dynamism.* |
| High ($> 0.66$) | High ($> 0$) | Dynamic attention is genuinely needed and delivers gains. GATv2 is the right choice. |
| High ($> 0.66$) | $\approx 0$ | GATv2 routes differently but gains nothing — possible overfitting or task insensitivity to routing. |

### Cross-graph workflow

When models are trained on graph $G_1$ and evaluated on graph $G_2$:

```text
Train GAT, GATv2 on G1  (freeze weights)
           ↓
Evaluate on G2:
  ├── compute DD_proxy  (Spearman, no labels needed)   ← structural signal
  └── compute Δacc      (task performance on G2)       ← functional signal
           ↓
Check: does high DD_proxy predict high Δacc across multiple G2 graphs?
```

This cross-graph correlation — $DD_{\text{proxy}}$ vs $\Delta\text{acc}$ across benchmark graphs — is a direct validation result for RQ2: it confirms that $DD_{\text{proxy}}$ is not just an architectural artefact but is predictive of when GATv2 will outperform GAT on unseen graphs.

> **Transfer caveat:** Freezing weights trained on $G_1$ and evaluating on $G_2$ assumes that the learned representations transfer well enough for attention distributions to remain meaningful. This holds when $G_1$ and $G_2$ are drawn from the same domain (e.g., two citation graphs). For heterogeneous benchmark suites, $DD_{\text{proxy}}$ should instead be computed from models trained directly on each target graph.

### Demand level thresholds

| $DD_{\text{proxy}}$ range | Label | Implication |
| --- | --- | --- |
| $[0,\ 0.33)$ | Low | Use GAT or a non-attention model |
| $[0.33,\ 0.66)$ | Moderate | GATv2 may help; check $\Delta\text{acc}$ |
| $[0.66,\ 1]$ | High | GATv2 is likely the better choice |

These thresholds are provisional. The cross-graph $DD_{\text{proxy}}$ vs $\Delta\text{acc}$ analysis in RQ2 will empirically calibrate which DD ranges correspond to significant accuracy gains, replacing these uniform cuts with data-driven boundaries.
