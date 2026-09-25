import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

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


def _cap(hits, max_per_s1):
    return set(sorted(hits)[:max_per_s1]) if len(hits) > max_per_s1 else hits

def candidates_for_row(name_norm, name_compact, addr_norm, country, idx, rules, min_name_len=3, min_addr_len=2, max_per_s1=200):
    hits = set()
    if "exact_name" in rules and name_norm:
        hits.update(idx.exact_name.get(name_norm, ()))
        if len(hits) >= max_per_s1:
            return _cap(hits, max_per_s1)
    if "exact_compact" in rules and name_compact:
        hits.update(idx.exact_compact.get(name_compact, ()))
        if len(hits) >= max_per_s1:
            return _cap(hits, max_per_s1)
    if "name_token" in rules:
        d = idx.name_tok
        for t in meaningful_name_tokens(name_norm, min_name_len):
            hits.update(d.get(t, ()))
            if len(hits) >= max_per_s1:
                return _cap(hits, max_per_s1)
    if "addr_token" in rules:
        d = idx.addr_tok
        for t in meaningful_addr_tokens(addr_norm, min_addr_len):
            hits.update(d.get(t, ()))
            if len(hits) >= max_per_s1:
                return _cap(hits, max_per_s1)
    if "country_name" in rules and name_norm:
        hits.update(idx.country_name.get((country, name_norm.split(" ")[0]), ()))
        if len(hits) >= max_per_s1:
            return _cap(hits, max_per_s1)
    if "country_addr" in rules and addr_norm:
        hits.update(idx.country_addr.get((country, addr_norm.split(" ")[0]), ()))
    return _cap(hits, max_per_s1)


def batch_rows(batch, idx2, idx3, rules, min_name_len, min_addr_len, max_per_s1):
    rows = []
    for row in batch.itertuples(index=False):
        s1_id = row.entity_id
        nn = row.name_norm if isinstance(row.name_norm, str) else ""
        nc = row.name_compact if isinstance(row.name_compact, str) else ""
        an = row.addr_norm if isinstance(row.addr_norm, str) else ""
        co = norm_country(row.country)
        h2 = candidates_for_row(nn, nc, an, co, idx2, rules,
                                min_name_len, min_addr_len, max_per_s1)
        h3 = candidates_for_row(nn, nc, an, co, idx3, rules,
                                min_name_len, min_addr_len, max_per_s1)
        cands = sorted({idx2.ids[p] for p in h2} | {idx3.ids[p] for p in h3})
        rows.append((s1_id, cands))
    return rows

def load_normalized(path, label):
    print(f"loading {label}: {path}", flush=True)
    df = pd.read_csv(path, sep="\t", dtype="string", keep_default_na=True)
    df = add_norm_columns(df)
    print(f"  rows={len(df):,}", flush=True)
    return df


def generate_candidate_pairs(s1_path, s2_path, s3_path, output_path, batch_size=25000, max_block_size=5000, max_per_s1=200, min_name_len=3, min_addr_len=2, rules=ALL_RULES, limit_s1=None):
    rules = tuple(r for r in rules if r in ALL_RULES)
    print(f"rules: {list(rules)}", flush=True)

    df2 = load_normalized(s2_path, "S2")
    idx2 = build_index(df2, "S2", min_name_len, min_addr_len, max_block_size)
    del df2
    gc.collect()

    df3 = load_normalized(s3_path, "S3")
    idx3 = build_index(df3, "S3", min_name_len, min_addr_len, max_block_size)
    del df3
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
            for s1_id, cands in batch_rows(chunk, idx2, idx3, rules,
                                           min_name_len, min_addr_len, max_per_s1):
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
