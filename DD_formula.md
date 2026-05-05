# Dynamism Demand — Formula Reference

---

## 1. DD_proxy(G) — Operational Definition

**Question answered:** Does using GATv2 over GAT change attention routing on this graph?
This is the primary target variable for RQ2 and RQ3.

### Step 1 — Per-node, per-layer, per-head Spearman correlation

For each node $i$ with $\deg(i) \geq 3$, layer $l$, and head $h$:

$$\rho_i^{(l,h)} = \text{Spearman}\!\left(\alpha_{\text{GAT}}^{(l,h)}[N(i)],\; \alpha_{\text{GATv2}}^{(l,h)}[N(i)]\right)$$

where $N(i)$ is the set of neighbours of node $i$ (including the self-loop added by GATConv), and $\alpha^{(l,h)}[N(i)]$ is the vector of attention weights node $i$ assigns to its neighbours at layer $l$, head $h$.

Nodes with $\deg(i) \leq 2$ are excluded not because Spearman is undefined, but because rank correlations computed over very few values are statistically brittle — they can only take values such as $\pm 1$ or $0$ and carry no statistical power. We retain only nodes with $\deg(i) \geq 3$, which gives at least four ranked positions (three real neighbours plus the self-loop). Four ranks is the minimum at which Spearman can produce values other than $\pm 1$ or $0$, making the estimates meaningfully graded rather than binary.

### Step 2 — Head-weighted aggregation across layers

$$\bar{\rho}_i = \frac{\displaystyle\sum_{l=1}^{L}\sum_{h=1}^{H_l} \rho_i^{(l,h)}}{\displaystyle\sum_{l=1}^{L} H_l}$$

This formula gives each individual attention head equal weight. A layer with more heads naturally contributes more to $\bar\rho_i$, but this is a consequence of head count — not a deliberate privileging of earlier layers. For the default 2-layer model ($H_0 = 2$, $H_1 = 1$), with heads indexed as $h \in \{1, 2\}$ for layer 0 and $h = 1$ for layer 1:

$$\bar{\rho}_i = \frac{\rho_i^{(0,1)} + \rho_i^{(0,2)} + \rho_i^{(1,1)}}{3}$$

This directly expands the general formula and avoids any ambiguity between per-head and per-layer quantities.

### Step 3 — Per-node dynamism demand

$$\text{dd}(i) = 1 - \max\!\left(0,\; \bar{\rho}_i\right) \;\in\; [0, 1]$$

Clamping at zero ensures anticorrelated rankings ($\bar\rho_i < 0$) do not push $\text{dd}(i)$ above 1 — anticorrelation is treated as equally demanding as zero correlation.

### Step 4 — Graph-level DD (degree-weighted mean)

$$\boxed{DD_{\text{proxy}}(G) = \frac{\displaystyle\sum_{\substack{i \in V \\ \deg(i)\,\geq\,3}} \deg(i)\cdot \text{dd}(i)}{\displaystyle\sum_{\substack{i \in V \\ \deg(i)\,\geq\,3}} \deg(i)}}$$

High-degree nodes receive greater weight because they have more reliable Spearman estimates and greater influence on message passing.

### Boundary conditions

| Scenario | $\bar\rho_i$ | $\text{dd}(i)$ | $DD_{\text{proxy}}$ |
| --- | --- | --- | --- |
| GAT $\equiv$ GATv2 everywhere | $1.0$ | $0.0$ | **0.0** — static attention suffices |
| Rankings fully uncorrelated | $0.0$ | $1.0$ | **1.0** — maximum dynamism demand |
| Rankings anticorrelated | $-1.0$ | $1.0$ (clamped) | **1.0** — treated as max demand |
| Partially correlated | $(0, 1)$ | $(0, 1)$ | **(0, 1)** — interpolated |

### Proxy validity assumption

$DD_{\text{proxy}}$ is sound under the assumption that **GAT learns the best static approximation** to the true attention function. Under this assumption:

- If all query nodes agree on neighbour rankings → static attention is optimal → GAT captures it → GATv2 does not deviate → $DD_{\text{proxy}} \approx 0$
- If query nodes strongly disagree → static attention fails → GATv2 diverges from GAT → $DD_{\text{proxy}} \approx 1$

The near-perfect cosine similarity ($\approx 0.999$) observed in RQ1 on ogbl-citation2 supports this assumption for well-trained models on citation graphs.

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
