"""
Five Dynamism Demand (DD) definitions — pure functions, no I/O.

All functions operate on an attention dict built from a (sub)graph's
head-averaged attention weights. Input and output are plain Python / numpy
scalars; torch tensors are accepted but converted internally.

Notation (from Brody et al. 2022 and the DD research plan):
    alpha_ij    : GATv2 attention weight, query i, key j
    N_i         : neighbourhood of node i (in the subgraph)
    S_{ii'}     : N_i ∩ N_{i'}, shared neighbours of i and i'

Pair-based definitions (RD, CQ, TKI) restrict to pairs with |S_{ii'}| >= 2
(or >= k for TKI), weighted by |S_{ii'}| for RD-DD.
"""

import math
from itertools import combinations

import numpy as np
import torch
from scipy.stats import kendalltau


# ---------------------------------------------------------------------------
# Adjacency dict construction
# ---------------------------------------------------------------------------

def build_attn_dict(sub_edge_index, alpha):
    """
    Build a per-query attention dict from edge-level tensors.

    Parameters
    ----------
    sub_edge_index : (2, E_sub) tensor — local node indices, row 0 = source/query
    alpha          : (E_sub,) tensor — head-averaged attention per edge

    Returns
    -------
    attn_dict : dict[int, dict[int, float]]
                {query_node: {neighbor: attention_weight}}
    """
    src = sub_edge_index[0].tolist()
    dst = sub_edge_index[1].tolist()
    a   = alpha.tolist()

    attn_dict = {}
    for s, d, w in zip(src, dst, a):
        attn_dict.setdefault(s, {})[d] = w
    return attn_dict


# ---------------------------------------------------------------------------
# Shared-neighbor pair enumeration
# ---------------------------------------------------------------------------

def _candidate_pairs(attn_dict):
    """
    Return all (i, i') pairs that share at least one common neighbour.

    Strategy: invert the adjacency — for each node c, every pair of nodes
    that both have c as a neighbour share c as a common neighbour.  This is
    O(sum_c  deg(c)^2), tractable for ego-subgraphs.

    Returns
    -------
    pairs : dict[(i, i'), frozenset] — shared neighbour set per pair
    """
    # Build reverse map: key → set of query nodes that have key as neighbour
    incoming = {}
    for i, nbrs in attn_dict.items():
        for j in nbrs:
            incoming.setdefault(j, set()).add(i)

    # For each common neighbour c, add all pairs (i, i') that both point to c
    shared = {}
    for c, queries in incoming.items():
        for i, ip in combinations(sorted(queries), 2):
            key = (i, ip)
            shared.setdefault(key, set()).add(c)

    return shared


# ---------------------------------------------------------------------------
# DD1 — Rank Disagreement (RD-DD)
# ---------------------------------------------------------------------------

def rd_dd(attn_dict):
    """
    Rank Disagreement DD.

    For each pair (i, i') with |S_{ii'}| >= 2:
        tau = Kendall's tau between ranked attention weights over S_{ii'}
    RD-DD = 1 - weighted_mean(tau, weight=|S_{ii'}|)

    Range [0, 1]: 0 = fully static (identical rankings), 1 = maximally dynamic.
    """
    pairs = _candidate_pairs(attn_dict)

    total_weight = 0.0
    weighted_tau = 0.0

    for (i, ip), shared in pairs.items():
        if len(shared) < 2:
            continue
        s_list = sorted(shared)
        a_i  = [attn_dict[i].get(j,  0.0) for j in s_list]
        a_ip = [attn_dict[ip].get(j, 0.0) for j in s_list]
        tau = kendalltau(a_i, a_ip).statistic
        if math.isnan(tau):
            continue
        w = len(shared)
        weighted_tau  += w * tau
        total_weight  += w

    if total_weight == 0:
        return float("nan")
    # tau in [-1, 1]; 1-tau in [0, 2]. Clamp to [0,1]: tau<0 means reversed
    # rankings, which is already maximally dynamic.
    raw = 1.0 - weighted_tau / total_weight
    return float(min(1.0, max(0.0, raw)))


# ---------------------------------------------------------------------------
# DD2 — Cross-Query Distribution Divergence (CQ-DD)
# ---------------------------------------------------------------------------

def _js_divergence(p, q, eps=1e-10):
    """Jensen-Shannon divergence in nats. Both p and q must sum to > 0."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * kl_pm + 0.5 * kl_qm


def cq_dd(attn_dict, eps=1e-10):
    """
    Cross-Query Distribution Divergence.

    For each pair (i, i') with |S_{ii'}| >= 2:
        Renormalise alpha_i and alpha_{i'} over S_{ii'} → p_i, p_{i'}
        JS = Jensen-Shannon divergence between p_i and p_{i'}
    CQ-DD = mean(JS / log(2))   — normalised to [0, 1].
    """
    pairs = _candidate_pairs(attn_dict)

    js_values = []
    for (i, ip), shared in pairs.items():
        if len(shared) < 2:
            continue
        s_list = sorted(shared)
        a_i  = np.array([attn_dict[i].get(j,  0.0) for j in s_list]) + eps
        a_ip = np.array([attn_dict[ip].get(j, 0.0) for j in s_list]) + eps
        js = _js_divergence(a_i, a_ip, eps=0.0)   # eps already added above
        js_values.append(js / math.log(2))

    if not js_values:
        return float("nan")
    return float(np.mean(js_values))


# ---------------------------------------------------------------------------
# DD3 — Dynamic Attention Advantage (GG-DD, redesigned)
# ---------------------------------------------------------------------------

def gg_dd(attn_v2_dict, attn_gat_dict, eps=1e-10):
    """
    Dynamic Attention Advantage (DAA) — CPU reference implementation.

    For each pair (i, i') with |S_{ii'}| >= 2:
        JSD_v2  = JS(alpha_v2_i  || alpha_v2_i')  over S_{ii'}
        JSD_gat = JS(alpha_gat_i || alpha_gat_i') over S_{ii'}
        advantage_ii' = JSD_v2 - JSD_gat
    GG-DD = mean(advantage_ii') / log(2)

    Positive: GATv2 produces more query-specific distributions than GAT.
    Zero: equivalent cross-query divergence (GAT is "effectively dynamic").
    Negative: degenerate; GAT is more query-specific on this subgraph.

    Natural zero baseline; does not saturate; measures whether GATv2 actually
    exploits its extra expressiveness relative to GAT's static baseline.
    """
    pairs_v2  = _candidate_pairs(attn_v2_dict)
    pairs_gat = _candidate_pairs(attn_gat_dict)
    common_pairs = set(pairs_v2) & set(pairs_gat)
    if not common_pairs:
        return float("nan")

    advantages = []
    for key in common_pairs:
        shared = pairs_v2[key] & pairs_gat[key]
        if len(shared) < 2:
            continue
        i, ip = key
        s_list = sorted(shared)
        a_v2_i  = np.array([attn_v2_dict[i].get(j,  0.0) for j in s_list]) + eps
        a_v2_ip = np.array([attn_v2_dict[ip].get(j, 0.0) for j in s_list]) + eps
        a_gat_i  = np.array([attn_gat_dict[i].get(j,  0.0) for j in s_list]) + eps
        a_gat_ip = np.array([attn_gat_dict[ip].get(j, 0.0) for j in s_list]) + eps
        jsd_v2  = _js_divergence(a_v2_i,  a_v2_ip,  eps=0.0) / math.log(2)
        jsd_gat = _js_divergence(a_gat_i, a_gat_ip, eps=0.0) / math.log(2)
        advantages.append(jsd_v2 - jsd_gat)

    if not advantages:
        return float("nan")
    return float(np.mean(advantages))


# ---------------------------------------------------------------------------
# DD4 — Entropy Lift (AEV-DD, redesigned)
# ---------------------------------------------------------------------------

def aev_dd(attn_v2_dict, attn_gat_dict, eps=1e-10):
    """
    Entropy Lift (EL-DD) — CPU reference implementation.

    For each node i with |N_i| >= 2 present in both dicts:
        H_norm_v2(i)  = H(alpha_v2_i)  / log(|N_i|)   in [0, 1]
        H_norm_gat(i) = H(alpha_gat_i) / log(|N_i|)   in [0, 1]
    AEV-DD = E_i[ H_norm_gat(i) - H_norm_v2(i) ]

    Positive when GATv2 is more concentrated (focused) than GAT — it attends
    more selectively, exploiting dynamic attention capacity. Zero when both
    models are equally diffuse. Negative if GATv2 spreads attention more than GAT.
    """
    common = set(attn_v2_dict) & set(attn_gat_dict)
    lifts = []
    for i in common:
        nbrs_v2  = attn_v2_dict[i]
        nbrs_gat = attn_gat_dict[i]
        shared = set(nbrs_v2) & set(nbrs_gat)
        if len(shared) < 2:
            continue
        log_n = math.log(len(shared))

        def _hnorm(nbrs, _shared=shared, _log_n=log_n):
            alpha = np.array([nbrs[j] for j in _shared], dtype=np.float64)
            alpha = np.clip(alpha, 0.0, None)
            h = -np.sum(alpha * np.log(alpha + eps))
            return h / _log_n

        lifts.append(_hnorm(nbrs_gat) - _hnorm(nbrs_v2))

    if not lifts:
        return float("nan")
    return float(np.mean(lifts))


# ---------------------------------------------------------------------------
# DD5 — Top-k Instability (TKI-DD)
# ---------------------------------------------------------------------------

def _compute_cq_gpu(a_l, a_r, cpid, counts, num_pairs, pair_chunk, device):
    """
    Compute CQ-DD scalar from a pre-built pair tensor.
    Extracted so gg_dd_gpu can call it twice (once per model) without
    duplicating the normalisation + JSD accumulation logic.
    """
    chunks = _pair_chunks(counts, pair_chunk, device)

    sum_l = torch.zeros(num_pairs, device=device)
    sum_r = torch.zeros(num_pairs, device=device)
    for _, _, es, ee in chunks:
        sum_l.scatter_add_(0, cpid[es:ee], a_l[es:ee])
        sum_r.scatter_add_(0, cpid[es:ee], a_r[es:ee])

    js_per_pair = torch.zeros(num_pairs, device=device)
    for _, _, es, ee in chunks:
        c  = cpid[es:ee]
        pl = a_l[es:ee] / sum_l[c]
        pr = a_r[es:ee] / sum_r[c]
        m  = 0.5 * (pl + pr)
        js = 0.5 * (pl * (pl.log() - m.log()) + pr * (pr.log() - m.log()))
        js_per_pair.scatter_add_(0, c, js)

    del sum_l, sum_r
    valid2 = counts >= 2
    if not valid2.any():
        return float("nan")
    return float((js_per_pair[valid2] / math.log(2)).mean().item())


def aev_dd_gpu(sub_edge_index, alpha_v2, alpha_gat, num_nodes, eps=1e-10):
    """
    Entropy Lift (EL-DD) — GPU-native via scatter_add.

    For each node v with |N_v| >= 2:
        H_norm_v2(v)  = H(alpha_v2_v)  / log(|N_v|)
        H_norm_gat(v) = H(alpha_gat_v) / log(|N_v|)
    AEV-DD = E_v[ H_norm_gat(v) - H_norm_v2(v) ]

    Positive when GATv2 is more concentrated (focused) than GAT — exploiting
    its dynamic attention capacity. Zero when both equally diffuse.
    Range roughly (-1, 1); expected positive on graphs where attention helps.
    """
    src    = sub_edge_index[0]
    device = alpha_v2.device

    degree = torch.zeros(num_nodes, device=device)
    degree.scatter_add_(0, src, torch.ones(src.size(0), device=device))
    valid = degree >= 2
    if valid.sum() < 1:
        return float("nan")

    def _hnorm(alpha):
        a = alpha.clamp(min=0.0)
        h_terms = -(a * torch.log(a + eps))
        h = torch.zeros(num_nodes, device=device)
        h.scatter_add_(0, src, h_terms)
        return (h[valid] / torch.log(degree[valid])).clamp(0.0, 1.0)

    hn_v2  = _hnorm(alpha_v2)
    hn_gat = _hnorm(alpha_gat)
    return float((hn_gat - hn_v2).mean().item())


def gg_dd_gpu(sub_edge_index, alpha_v2, alpha_gat, num_nodes,
              pair_chunk=0, neighbor_cap=0, eps=1e-10):
    """
    Dynamic Attention Advantage (DAA-DD) — GPU-native.

    DAA-DD = CQ-DD(GATv2) - CQ-DD(GAT)

    For each pair (i, i') with |S_{ii'}| >= 2, the per-pair JSD is computed
    independently for GATv2 and GAT attention, then averaged. The difference
    of averages equals the average difference (linearity of expectation).

    Positive: GATv2 is more query-specific than GAT across shared neighborhoods.
    Zero:     equivalent cross-query divergence (GAT is "effectively dynamic").
    Negative: GAT is more query-specific; GATv2's dynamic capacity is unexploited.

    Uses _build_pair_tensors_gpu twice (once per model) sharing no intermediate
    tensors, so peak VRAM is bounded by one pair tensor at a time.
    """
    # CQ-DD for GATv2
    res_v2 = _build_pair_tensors_gpu(sub_edge_index, alpha_v2, num_nodes,
                                      neighbor_cap=neighbor_cap, eps=eps)
    if res_v2 is None:
        return float("nan")
    a_l_v2, a_r_v2, cpid_v2, counts_v2, np_v2 = res_v2
    device = a_l_v2.device
    cq_v2 = _compute_cq_gpu(a_l_v2, a_r_v2, cpid_v2, counts_v2, np_v2,
                              pair_chunk, device)
    del a_l_v2, a_r_v2, cpid_v2, counts_v2

    # CQ-DD for GAT
    res_gat = _build_pair_tensors_gpu(sub_edge_index, alpha_gat, num_nodes,
                                       neighbor_cap=neighbor_cap, eps=eps)
    if res_gat is None:
        return float("nan")
    a_l_gat, a_r_gat, cpid_gat, counts_gat, np_gat = res_gat
    cq_gat = _compute_cq_gpu(a_l_gat, a_r_gat, cpid_gat, counts_gat, np_gat,
                               pair_chunk, device)
    del a_l_gat, a_r_gat, cpid_gat, counts_gat

    if math.isnan(cq_v2) or math.isnan(cq_gat):
        return float("nan")
    return cq_v2 - cq_gat


# ---------------------------------------------------------------------------
# DD5 — Top-k Instability (TKI-DD)
# ---------------------------------------------------------------------------

def tki_dd(attn_dict, k_list=(1, 3, 5)):
    """
    Top-k Instability DD.

    For each pair (i, i') with |S_{ii'}| >= k:
        T_k(i)|_S  = top-k neighbours of i in S_{ii'} by attention weight
        Jaccard_dissim = 1 - |T_k(i)|_S ∩ T_k(i')|_S| / |T_k(i)|_S ∪ T_k(i')|_S|
    TKI-DD_k = mean(Jaccard_dissim) over all valid pairs.

    Returns dict {k: float for k in k_list}.
    Primary metric is k=1 (most conservative; tied to DictionaryLookup failure).
    """
    pairs = _candidate_pairs(attn_dict)
    k_list = sorted(set(k_list))

    accum  = {k: [] for k in k_list}

    for (i, ip), shared in pairs.items():
        s_list = sorted(shared)
        for k in k_list:
            if len(s_list) < k:
                continue
            # top-k from shared neighbours, by GATv2 attention of each query
            def topk_set(query):
                weights = [(attn_dict[query].get(j, 0.0), j) for j in s_list]
                weights.sort(reverse=True)
                return {j for _, j in weights[:k]}

            t_i  = topk_set(i)
            t_ip = topk_set(ip)

            intersection = len(t_i & t_ip)
            union        = len(t_i | t_ip)
            jaccard_dissim = 1.0 - intersection / union if union > 0 else 0.0
            accum[k].append(jaccard_dissim)

    result = {}
    for k in k_list:
        vals = accum[k]
        result[k] = float(np.mean(vals)) if vals else float("nan")
    return result


# ---------------------------------------------------------------------------
# GPU — shared pair-tensor infrastructure for CQ-DD and TKI-DD
# ---------------------------------------------------------------------------

def _build_pair_tensors_gpu(sub_edge_index, alpha, num_nodes, neighbor_cap=0, eps=1e-10):
    """
    For every pair of source nodes (i, i') sharing at least one common
    destination, build a flat contribution tensor sorted by pair.

    The algorithm:
      1. Sort edges by destination to group them by shared neighbour.
      2. Within each group of size d, generate all C(min(d,neighbor_cap),2)
         (left, right) edge pairs using repeat_interleave + offset arithmetic
         — no Python loop. neighbor_cap=0 means no cap (use all d sources).
      3. Assign a canonical int64 pair ID and sort so each pair's contributions
         are contiguous.

    neighbor_cap > 0 prevents quadratic blowup from hub nodes with very high
    in-degree (common in citation graphs): a node with d=5000 generates
    C(5000,2)=12.5M pairs; with neighbor_cap=50 it generates at most C(50,2)=1225.

    Returns
    -------
    a_l, a_r        : (P,) float32  clamped alpha for left / right source
    compact_pair_id : (P,) int64    0-based pair index per entry
    counts          : (num_pairs,) int64  shared-destination count per pair
    num_pairs       : int
    Returns None when no pairs exist.
    """
    src    = sub_edge_index[0]
    dst    = sub_edge_index[1]
    device = alpha.device
    E      = src.size(0)

    a = alpha.clamp(min=0.0) + eps

    # --- group edges by destination ---
    order  = dst.argsort(stable=True)
    dst_s  = dst[order]
    src_s  = src[order]
    a_s    = a[order]

    is_new     = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        dst_s[1:] != dst_s[:-1],
    ])
    seg_id     = is_new.cumsum(0) - 1                              # (E,)
    seg_starts = is_new.nonzero(as_tuple=False).squeeze(1)         # (G,)
    seg_ends   = torch.cat([seg_starts[1:], torch.tensor([E], device=device)])
    seg_size   = seg_ends - seg_starts                             # (G,)

    seg_size_e = seg_size[seg_id]                                  # (E,)
    if neighbor_cap > 0:
        seg_size_e = seg_size_e.clamp(max=neighbor_cap)
    local_pos  = torch.arange(E, device=device) - seg_starts[seg_id]
    # each edge at local position p generates (cap-1-p) right partners, 0 if p >= cap
    n_right_e  = (seg_size_e - 1 - local_pos).clamp(min=0)        # (E,)

    total = int(n_right_e.sum().item())
    if total == 0:
        return None

    # --- vectorised pair generation ---
    left_idx  = torch.repeat_interleave(torch.arange(E, device=device), n_right_e)
    cum_n     = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        n_right_e.cumsum(0),
    ])
    offset    = torch.arange(total, device=device) - cum_n[left_idx] + 1
    del cum_n
    right_idx = left_idx + offset
    del offset

    # Extract and immediately free index arrays
    src_l = src_s[left_idx]
    src_r = src_s[right_idx]
    a_l   = a_s[left_idx]
    a_r   = a_s[right_idx]
    del left_idx, right_idx

    # --- canonical pair ID: encode (min, max) as a single int64 ---
    p_min   = torch.minimum(src_l, src_r).to(torch.int64)
    p_max   = torch.maximum(src_l, src_r).to(torch.int64)
    del src_l, src_r
    pair_id = p_min * num_nodes + p_max
    del p_min, p_max

    # --- sort by pair_id to group contributions per pair ---
    sort_ord  = pair_id.argsort(stable=True)
    a_l       = a_l[sort_ord]
    a_r       = a_r[sort_ord]
    pair_id_s = pair_id[sort_ord]
    del pair_id, sort_ord

    is_new_pair     = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        pair_id_s[1:] != pair_id_s[:-1],
    ])
    del pair_id_s
    compact_pair_id = (is_new_pair.cumsum(0) - 1).to(torch.long)
    del is_new_pair
    num_pairs       = int(compact_pair_id[-1].item()) + 1

    counts = torch.zeros(num_pairs, dtype=torch.long, device=device)
    counts.scatter_add_(0, compact_pair_id,
                        torch.ones(total, dtype=torch.long, device=device))

    return a_l, a_r, compact_pair_id, counts, num_pairs


def _pair_chunks(counts, pair_chunk, device):
    """
    Partition pairs into batches of at most pair_chunk entries each,
    aligned to pair boundaries so no pair is split across chunks.

    Returns list of (p_start, p_end, e_start, e_end) tuples.
    pair_chunk <= 0 means one chunk containing everything.
    """
    num_pairs = counts.size(0)
    pair_entry_starts = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        counts.cumsum(0),
    ])
    total = int(pair_entry_starts[-1].item())

    if pair_chunk <= 0 or total <= pair_chunk:
        return [(0, num_pairs, 0, total)]

    # Assign each pair to the chunk whose window covers its first entry
    chunk_ids = pair_entry_starts[:-1] // pair_chunk
    is_new = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        chunk_ids[1:] != chunk_ids[:-1],
    ])
    p_starts = is_new.nonzero(as_tuple=False).squeeze(1)
    p_ends   = torch.cat([p_starts[1:], torch.tensor([num_pairs], device=device)])
    e_starts = pair_entry_starts[p_starts]
    e_ends   = pair_entry_starts[p_ends]

    return list(zip(p_starts.tolist(), p_ends.tolist(),
                    e_starts.tolist(), e_ends.tolist()))


def cq_tki_rd_dd_gpu(sub_edge_index, alpha, num_nodes,
                     k_list=(1, 3, 5), pair_chunk=0, neighbor_cap=0, eps=1e-10):
    """
    GPU-native CQ-DD, TKI-DD, and RD-DD in a single pass over the pair tensor.

    CQ-DD  — JSD between each pair's renormalised attention distributions over
              shared neighbours.
    TKI-DD — Jaccard dissimilarity of top-k attention sets over shared
              neighbours, for every k in k_list.
    RD-DD  — Exact Kendall's tau via a secondary neighbour-pair enumeration:
              for each query pair (i,i') the shared-neighbour entries are
              treated as a group, and all C(|S|,2) neighbour-neighbour pairs
              are enumerated with the same repeat_interleave trick used for
              the primary pair tensor.  This replaces the scipy per-call loop.

    All three metrics share one call to _build_pair_tensors_gpu.

    pair_chunk controls the max pair-destination entries processed per GPU
    batch. 0 = process all at once. Reduce to cut peak VRAM usage.

    neighbor_cap > 0 caps the number of sources considered per destination
    node before pair generation, bounding each node's pair contribution to
    C(neighbor_cap, 2). Essential for citation graphs with hub nodes.

    Returns
    -------
    cq  : float
    tki : dict {k: float}
    rd  : float
    """
    nan    = float("nan")
    k_list = sorted(set(k_list))

    result = _build_pair_tensors_gpu(sub_edge_index, alpha, num_nodes,
                                     neighbor_cap=neighbor_cap, eps=eps)
    if result is None:
        return nan, {k: nan for k in k_list}, nan

    a_l, a_r, cpid, counts, num_pairs = result
    device = a_l.device
    total  = a_l.size(0)

    chunks = _pair_chunks(counts, pair_chunk, device)

    # ------------------------------------------------------------------ CQ-DD
    valid2 = counts >= 2
    cq = _compute_cq_gpu(a_l, a_r, cpid, counts, num_pairs, pair_chunk, device)

    # ---------------------------------------------------------------- TKI-DD
    # Global sort to get within-pair ranks (unavoidable); sort key freed
    # immediately after argsort. Rank assignment and inter_count are chunked.
    pair_cumstart = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        counts.cumsum(0)[:-1],
    ])
    flat_pos = torch.arange(total, device=device)
    max_a    = float(max(a_l.max().item(), a_r.max().item())) + 1.0

    sort_key_l = cpid.double() * max_a - a_l.double()
    perm_l     = sort_key_l.argsort(stable=True)
    del sort_key_l

    rank_l = torch.empty(total, dtype=torch.long, device=device)
    for i in range(0, total, pair_chunk if pair_chunk > 0 else total):
        sl = perm_l[i : i + (pair_chunk if pair_chunk > 0 else total)]
        rank_l[sl] = flat_pos[i : sl.size(0) + i] - pair_cumstart[cpid[sl]]
    del perm_l

    sort_key_r = cpid.double() * max_a - a_r.double()
    perm_r     = sort_key_r.argsort(stable=True)
    del sort_key_r

    rank_r = torch.empty(total, dtype=torch.long, device=device)
    for i in range(0, total, pair_chunk if pair_chunk > 0 else total):
        sl = perm_r[i : i + (pair_chunk if pair_chunk > 0 else total)]
        rank_r[sl] = flat_pos[i : sl.size(0) + i] - pair_cumstart[cpid[sl]]
    del perm_r

    tki = {}
    for k in k_list:
        valid_k = counts >= k
        if not valid_k.any():
            tki[k] = nan
            continue

        inter_count = torch.zeros(num_pairs, dtype=torch.long, device=device)
        for _, _, es, ee in chunks:
            in_inter = (rank_l[es:ee] < k) & (rank_r[es:ee] < k)
            inter_count.scatter_add_(0, cpid[es:ee], in_inter.long())
            del in_inter

        union  = (2 * k - inter_count[valid_k]).clamp(min=1)
        dissim = 1.0 - inter_count[valid_k].float() / union.float()
        tki[k] = float(dissim.mean().item())
        del inter_count, union, dissim

    del rank_l, rank_r, pair_cumstart, flat_pos

    # ----------------------------------------------------------------- RD-DD
    # Chunked secondary neighbour-pair enumeration.
    # Each chunk processes a slice of primary entries (complete pairs only).
    # For each (j1,j2) within pair p: concordant = sign(Δa_l)*sign(Δa_r).
    if not valid2.any():
        return cq, tki, nan

    C_minus_D = torch.zeros(num_pairs, device=device)
    for ps, pe, es, ee in chunks:
        chunk_a_l   = a_l[es:ee]
        chunk_a_r   = a_r[es:ee]
        chunk_cpid  = cpid[es:ee] - ps          # local 0-based pair IDs
        chunk_counts = counts[ps:pe]
        chunk_n      = ee - es

        chunk_pc = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            chunk_counts.cumsum(0)[:-1],
        ])
        chunk_fp  = torch.arange(chunk_n, device=device)
        chunk_lp  = chunk_fp - chunk_pc[chunk_cpid]
        n_right   = (chunk_counts[chunk_cpid] - 1 - chunk_lp).clamp(min=0)
        del chunk_lp

        total2 = int(n_right.sum().item())
        if total2 == 0:
            del chunk_a_l, chunk_a_r, chunk_cpid, chunk_counts, chunk_pc, chunk_fp, n_right
            continue

        left2  = torch.repeat_interleave(chunk_fp, n_right)
        cum_n2 = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            n_right.cumsum(0),
        ])
        del chunk_fp, n_right
        off2   = torch.arange(total2, device=device) - cum_n2[left2] + 1
        del cum_n2
        right2 = left2 + off2
        del off2

        sign_l  = (chunk_a_l[left2] - chunk_a_l[right2]).sign()
        sign_r  = (chunk_a_r[left2] - chunk_a_r[right2]).sign()
        concord = sign_l * sign_r
        del sign_l, sign_r

        qpid = chunk_cpid[left2]
        del left2, right2, chunk_a_l, chunk_a_r, chunk_cpid

        local_cmd = torch.zeros(pe - ps, device=device)
        local_cmd.scatter_add_(0, qpid, concord.float())
        del qpid, concord, chunk_counts, chunk_pc

        C_minus_D[ps:pe].add_(local_cmd)
        del local_cmd

    n_nbr_pairs  = counts * (counts - 1) // 2
    tau_per_pair = C_minus_D[valid2] / n_nbr_pairs[valid2].float()
    weights      = counts[valid2].float()
    raw_rd       = 1.0 - float((tau_per_pair * weights).sum() / weights.sum())
    rd           = float(min(1.0, max(0.0, raw_rd)))

    return cq, tki, rd
