import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src import data_loader as dl
from src.normalization import add_norm_columns

CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]

ALL_RULES = (
    "exact_name",
    "exact_compact",
    "name_token",
    "addr_token",
    "country_name",
    "country_addr",
)

# Generic tokens that would create mega-blocks; skipped for token rules.
# Legal-suffix meaning is preserved via the exact-name rules above.
NAME_STOP = frozenset({
    "llc", "ltd", "inc", "corp", "co", "pvt", "plc", "llp", "pllc",
    "and", "the", "of", "a", "an", "de", "la",
})


def norm_country(s):
    if s is None:
        return ""
    try:
        if pd.isna(s):
            return ""
    except Exception:
        pass
    return str(s).strip().lower()


def meaningful_name_tokens(name_norm, min_len=3):
    if not name_norm:
        return []
    return [t for t in name_norm.split(" ") if len(t) >= min_len and t not in NAME_STOP]


def meaningful_addr_tokens(addr_norm, min_len=2):
    if not addr_norm:
        return []
    out = []
    for t in addr_norm.split(" "):
        if len(t) < min_len:
            continue
        if not any(ch.isalnum() for ch in t):
            continue
        out.append(t)
    return out


class BlockIndex:
    def __init__(self, ids):
        self.ids = ids     # list[str], position-aligned
        self.exact_name = {}
        self.exact_compact = {}
        self.name_tok = {}
        self.addr_tok = {}
        self.country_name = {}
        self.country_addr = {}

    @property
    def n(self):
        return len(self.ids)


def _add(index_dict, key, pos):
    if key:
        index_dict.setdefault(key, []).append(pos)


def _prune(d, cap, name, label):
    kept = {k: v for k, v in d.items() if len(v) <= cap}
    print(f"[{label}] {name}: keys={len(kept):,} "
          f"(dropped {len(d) - len(kept):,} over cap={cap})", flush=True)
    return kept


def build_index(df, label, min_name_len=3, min_addr_len=2, max_block_size=5000):
    ids = df["entity_id"].tolist()
    countries = [norm_country(c) for c in df["country"].tolist()]
    idx = BlockIndex(ids)

    en, ec, nt, at, cn, ca = ({}, {}, {}, {}, {}, {})
    for pos, (name_norm, name_compact, addr_norm) in enumerate(zip(
        df["name_norm"].tolist(), df["name_compact"].tolist(), df["addr_norm"].tolist()
    )):
        name_norm = name_norm if isinstance(name_norm, str) else ""
        name_compact = name_compact if isinstance(name_compact, str) else ""
        addr_norm = addr_norm if isinstance(addr_norm, str) else ""
        _add(en, name_norm, pos)
        _add(ec, name_compact, pos)
        for t in meaningful_name_tokens(name_norm, min_name_len):
            _add(nt, t, pos)
        for t in meaningful_addr_tokens(addr_norm, min_addr_len):
            _add(at, t, pos)
        c = countries[pos]
        if name_norm:
            _add(cn, (c, name_norm.split(" ")[0]), pos)
        if addr_norm:
            _add(ca, (c, addr_norm.split(" ")[0]), pos)

    idx.exact_name = _prune(en, max_block_size, "exact_name", label)
    idx.exact_compact = _prune(ec, max_block_size, "exact_compact", label)
    idx.name_tok = _prune(nt, max_block_size, "name_tok", label)
    idx.addr_tok = _prune(at, max_block_size, "addr_tok", label)
    idx.country_name = _prune(cn, max_block_size, "country_name", label)
    idx.country_addr = _prune(ca, max_block_size, "country_addr", label)
    gc.collect()
    return idx


def candidates_for_row(name_norm, name_compact, addr_norm, country, idx, rules, min_name_len=3, min_addr_len=2):
    hits = set()
    if "exact_name" in rules and name_norm:
        hits.update(idx.exact_name.get(name_norm, ()))
    if "exact_compact" in rules and name_compact:
        hits.update(idx.exact_compact.get(name_compact, ()))
    if "name_token" in rules:
        d = idx.name_tok
        for t in meaningful_name_tokens(name_norm, min_name_len):
            hits.update(d.get(t, ()))
    if "addr_token" in rules:
        d = idx.addr_tok
        for t in meaningful_addr_tokens(addr_norm, min_addr_len):
            hits.update(d.get(t, ()))
    if "country_name" in rules and name_norm:
        hits.update(idx.country_name.get((country, name_norm.split(" ")[0]), ()))
    if "country_addr" in rules and addr_norm:
        hits.update(idx.country_addr.get((country, addr_norm.split(" ")[0]), ()))
    return hits


TFIDF_ANALYZER = "char"
TFIDF_NGRAM_RANGE = (2, 5)


class TfidfSide:
    def __init__(self, vectorizer, matrix):
        self.vectorizer = vectorizer
        self.matrix = matrix  # csr, L2-normalized rows -> dot product == cosine

    @property
    def n(self):
        return self.matrix.shape[0]


def build_tfidf(names, label):
    """Fit char n-gram TF-IDF on one source's `name_norm` list. Keeps sparse matrix."""
    vec = TfidfVectorizer(analyzer=TFIDF_ANALYZER, ngram_range=TFIDF_NGRAM_RANGE,
                          lowercase=False)
    mat = vec.fit_transform(names).tocsr()
    print(f"[{label}] tfidf: docs={mat.shape[0]:,} features={mat.shape[1]:,} "
          f"nnz={mat.nnz:,}", flush=True)
    return TfidfSide(vec, mat)


def tfidf_topk_restricted(side, q_vec, cand_positions, top_k, min_similarity):
    """Memory-safe TF-IDF top-k restricted to a blocking candidate pool.

    Scores a single query row against only ``cand_positions`` rows of
    ``side.matrix``: ``q (1xF) @ Cand (CxF).T -> (1xC)``. The dense output
    is size C (pool size, manageable) instead of N (full side, huge).
    Returns candidate positions ordered by similarity desc.
    """
    if side is None or top_k <= 0 or not cand_positions:
        return []
    if q_vec is None or q_vec.nnz == 0:
        return []
    cand_list = sorted(cand_positions)  # deterministic row alignment
    sub = side.matrix[cand_list]  # C x F, transient per-row copy
    scores = (q_vec @ sub.T).toarray().ravel()  # dense C
    mask = scores >= min_similarity
    if not np.any(mask):
        return []
    idx = np.flatnonzero(mask)
    if len(idx) > top_k:
        part = np.argpartition(-scores[idx], top_k - 1)[:top_k]
        part = part[np.argsort(-scores[idx][part], kind="stable")]
        idx = idx[part]
    else:
        idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [cand_list[j] for j in idx.tolist()]


def batch_rows(batch, idx2, idx3, rules, min_name_len, min_addr_len, max_per_s1, tf2=None, tf3=None, tfidf_top_k=20, tfidf_min_similarity=0.55, tfidf_query_batch=2000):
    names = [n if isinstance(n, str) else "" for n in batch["name_norm"].tolist()]
    # Materialize row fields once so sub-batching below stays aligned.
    recs = []
    for row in batch.itertuples(index=False):
        recs.append((
            row.entity_id,
            row.name_norm if isinstance(row.name_norm, str) else "",
            row.name_compact if isinstance(row.name_compact, str) else "",
            row.addr_norm if isinstance(row.addr_norm, str) else "",
            norm_country(row.country),
        ))
    rows = [None] * len(recs)
    use_tfidf = (tf2 is not None or tf3 is not None) and tfidf_top_k > 0
    qb = max(1, int(tfidf_query_batch))
    for start in range(0, len(recs), qb):
        end = min(len(recs), start + qb)
        sub_names = names[start:end]
        Q2 = tf2.vectorizer.transform(sub_names) if tf2 is not None else None
        Q3 = tf3.vectorizer.transform(sub_names) if tf3 is not None else None
        for j in range(start, end):
            s1_id, nn, nc, an, co = recs[j]
            h2 = candidates_for_row(nn, nc, an, co, idx2, rules, min_name_len, min_addr_len)
            h3 = candidates_for_row(nn, nc, an, co, idx3, rules, min_name_len, min_addr_len)
            all_ids = {idx2.ids[p] for p in h2} | {idx3.ids[p] for p in h3}
            if use_tfidf and all_ids:
                k = j - start
                # TF-IDF scored ONLY within the blocking pools (no global Q@T).
                t2 = tfidf_topk_restricted(tf2, Q2[k], h2, tfidf_top_k, tfidf_min_similarity) if tf2 is not None else []
                t3 = tfidf_topk_restricted(tf3, Q3[k], h3, tfidf_top_k, tfidf_min_similarity) if tf3 is not None else []
                if t2 or t3:
                    # Priority (similarity-ordered) used when truncating to max_per_s1.
                    prio, seen = [], set()
                    for p in t2:
                        eid = idx2.ids[p]
                        if eid not in seen:
                            seen.add(eid)
                            prio.append(eid)
                    for p in t3:
                        eid = idx3.ids[p]
                        if eid not in seen:
                            seen.add(eid)
                            prio.append(eid)
                    if len(all_ids) > max_per_s1:
                        rest = sorted(all_ids - seen)
                        rows[j] = (s1_id, (prio + rest)[:max_per_s1])
                        continue
            cands = sorted(all_ids)
            if len(cands) > max_per_s1:
                # No TF-IDF priority available: deterministic sorted truncation.
                # With TF-IDF priority available this branch is unreachable
                # (handled above), keeping blocking recall otherwise.
                cands = cands[:max_per_s1]
            rows[j] = (s1_id, cands)
        del Q2, Q3
        gc.collect()
    return rows

def load_normalized(path, label):
    print(f"loading {label}: {path}", flush=True)
    df = pd.read_csv(path, sep="\t", dtype="string", keep_default_na=True)
    df = add_norm_columns(df)
    print(f"  rows={len(df):,}", flush=True)
    return df


def generate_candidate_pairs(s1_path, s2_path, s3_path, output_path, batch_size=25000, max_block_size=5000, max_per_s1=200, min_name_len=3, min_addr_len=2, rules=ALL_RULES, limit_s1=None, use_tfidf=True, tfidf_top_k=20, tfidf_min_similarity=0.55, tfidf_query_batch=2000):
    rules = tuple(r for r in rules if r in ALL_RULES)
    print(f"rules: {list(rules)} | tfidf={use_tfidf} " f"(top_k={tfidf_top_k}, min_sim={tfidf_min_similarity})", flush=True)

    df2 = load_normalized(s2_path, "S2")
    idx2 = build_index(df2, "S2", min_name_len, min_addr_len, max_block_size)
    names2 = [n if isinstance(n, str) else "" for n in df2["name_norm"].tolist()]
    del df2
    gc.collect()
    tf2 = build_tfidf(names2, "S2") if use_tfidf else None
    del names2
    gc.collect()

    df3 = load_normalized(s3_path, "S3")
    idx3 = build_index(df3, "S3", min_name_len, min_addr_len, max_block_size)
    names3 = [n if isinstance(n, str) else "" for n in df3["name_norm"].tolist()]
    del df3
    gc.collect()
    tf3 = build_tfidf(names3, "S3") if use_tfidf else None
    del names3
    gc.collect()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()  # fresh reproducible run

    total_rows, total_s1 = 0, 0
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        reader = pd.read_csv(s1_path, sep="\t", dtype="string",
                             keep_default_na=True, chunksize=batch_size)
        for chunk in reader:
            if limit_s1 is not None and total_s1 >= limit_s1:
                break
            if limit_s1 is not None:
                chunk = chunk.iloc[: max(0, limit_s1 - total_s1)]
                if len(chunk) == 0:
                    break
            chunk = add_norm_columns(chunk)
            for s1_id, cands in batch_rows(chunk, idx2, idx3, rules, min_name_len, min_addr_len, max_per_s1, tf2, tf3, tfidf_top_k, tfidf_min_similarity, tfidf_query_batch):
                f.write(f"{s1_id}\t{','.join(cands)}\n")
                total_rows += 1
            total_s1 += len(chunk)
            print(f"S1 processed={total_s1:,} rows={total_rows:,}", flush=True)
            del chunk
            gc.collect()

    print(f"DONE s1={total_s1:,} rows={total_rows:,} -> {output_path}", flush=True)
    return output_path


def generate_train_candidate_pairs(output_path, batch_size=25000, **kwargs):
    return generate_candidate_pairs(dl.TRAIN_S1, dl.TRAIN_S2, dl.TRAIN_S3, output_path, batch_size=batch_size, **kwargs)


def generate_test_candidate_pairs(output_path, batch_size=25000, **kwargs):
    return generate_candidate_pairs(dl.TEST_S1, dl.TEST_S2, dl.TEST_S3, output_path, batch_size=batch_size, **kwargs)
