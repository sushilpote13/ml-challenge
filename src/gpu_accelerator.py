import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_device():
    print(f"[gpu_accelerator] torch={torch.__version__} "f"cuda_available={torch.cuda.is_available()} device={DEVICE}", flush=True)
    return DEVICE

def csr_to_torch_sparse(csr, device=None):
    """scipy CSR (float) -> torch sparse_csr_tensor on ``device`` (transfer format)."""
    import warnings
    csr = csr.tocsr()
    crow = torch.tensor(csr.indptr.astype(np.int64), dtype=torch.int64)
    cols = torch.tensor(csr.indices.astype(np.int64), dtype=torch.int64)
    vals = torch.tensor(csr.data.astype(np.float64), dtype=torch.float64)
    with warnings.catch_warnings():
        # Input comes straight from scipy (valid invariants); silence the
        # beta/invariant construction warnings for clean pipeline logs.
        warnings.simplefilter("ignore")
        return torch.sparse_csr_tensor(crow, cols, vals, size=csr.shape, device=device or DEVICE)

def _row_nnz(csr, i):
    return csr.indptr[i + 1] - csr.indptr[i]

def score_candidates_batch(Q_csr, M_csr, row_pools, top_k, min_similarity,
                           device=None, group_size=256, max_union=4000,
                           max_features=4000):
    B = Q_csr.shape[0]
    out = [[] for _ in range(B)]
    if top_k <= 0 or B == 0:
        return out
    dev = device or DEVICE
    # Worst-case dense bound for one group: (G*Fg + U*Fg + G*U) float32.
    bound_mb = (group_size * max_features + max_union * max_features
                + group_size * max_union) * 4 / 1e6
    print(f"[gpu_accelerator] scoring {B} rows on {dev} "
          f"(group<={group_size}, union<={max_union}, feats<={max_features}, "
          f"worst-case ~{bound_mb:,.1f}MB/group fp32)", flush=True)
    pools = [sorted(set(p)) for p in row_pools]

    # Group rows (input order) so each device call stays bounded.
    groups, cur_rows, cur_union = [], [], set()
    for i, pool in enumerate(pools):
        if not pool or _row_nnz(Q_csr, i) == 0:
            continue  # matches legacy: empty pool / zero query vector -> []
        grown = cur_union | set(pool)
        if cur_rows and (len(grown) > max_union or len(cur_rows) >= group_size):
            groups.append((cur_rows, sorted(cur_union)))
            cur_rows, cur_union = [], set()
            grown = set(pool)
        cur_rows.append(i)
        cur_union = grown
    if cur_rows:
        groups.append((cur_rows, sorted(cur_union)))

    with torch.no_grad():
        for rows_g, U in groups:
            G = len(rows_g)
            Qg = Q_csr[rows_g]
            Mu = M_csr[U]
            # Restrict features to those present in this group (exact dots,
            # small dense blocks). Split the group if still too wide.
            fset = set(Qg.indices.tolist()) | set(Mu.indices.tolist())
            if len(fset) > max_features:
                mid = G // 2 or 1
                for half in (rows_g[:mid], rows_g[mid:]):
                    sub = score_candidates_batch(
                        Q_csr[half], M_csr, [pools[i] for i in half],
                        top_k, min_similarity, dev,
                        group_size, max_union, max_features)
                    for i, hit in zip(half, sub):
                        out[i] = hit
                continue
            Fidx = sorted(fset)
            Qd = torch.tensor(Qg.tocsc()[:, Fidx].toarray(), dtype=torch.float32, device=dev)
            Md = torch.tensor(Mu.tocsc()[:, Fidx].toarray(), dtype=torch.float32, device=dev)
            S = (Qd @ Md.T).to("cpu").numpy()
            del Qd, Md
            U_arr = np.asarray(U)
            for gi, i in enumerate(rows_g):
                member = np.isin(U_arr, pools[i])
                s = np.where(member, S[gi], -np.inf)
                keep = np.flatnonzero(s >= min_similarity)
                if keep.size == 0:
                    continue
                # (score desc, position asc) — matches legacy stable-desc
                # order on position-ascending CSR rows.
                order = np.lexsort((U_arr[keep], -s[keep]))
                keep = keep[order[:top_k]]
                out[i] = [int(U_arr[j]) for j in keep.tolist()]
            del S
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    return out
