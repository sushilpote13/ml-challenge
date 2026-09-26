import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src import data_loader as dl
from src.normalization import add_norm_columns
from src import similarity as sim

FEATURE_NAMES = [
    "name_exact", "name_compact_exact",
    "name_jaccard", "name_overlap_ratio", "name_levenshtein", "name_jaro_winkler",
    "name_containment", "name_first_token", "name_len_ratio",
    "addr_exact",
    "addr_jaccard", "addr_overlap_ratio", "addr_levenshtein", "addr_jaro_winkler",
    "addr_containment", "addr_len_ratio",
    "addr_digit_overlap", "addr_has_digits_both",
    "country_match",
]

_LOOKUP_COLS = ["name_norm", "addr_norm", "name_compact", "country"]

def _load_source_lookup(path, label):
    df = pd.read_csv(path, sep="\t", dtype="string", keep_default_na=True)
    df = add_norm_columns(df)
    df = df.set_index("entity_id")
    print(f"[{label}] loaded {len(df):,} rows for feature lookup", flush=True)
    return df

def _explode_candidates(candidate_pairs_path):
    df = pd.read_csv(candidate_pairs_path, sep="\t", dtype="string", keep_default_na=False)
    df = df[df["candidate_entity_ids"] != ""].copy()
    df["candidate_entity_id"] = df["candidate_entity_ids"].str.split(",")
    df = df.explode("candidate_entity_id")
    return df[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)

def _load_ground_truth_pairs(gt_path):
    gt = pd.read_csv(gt_path, sep="\t", dtype="string", keep_default_na=False)
    pairs = set()
    for s1_id, matches in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if not matches:
            continue
        for m in matches.split(","):
            pairs.add((s1_id, m))
    return pairs


def _attach_fields(chunk, s1_df, s2_df, s3_df):
    s1_fields = s1_df.reindex(chunk["source1_entity_id"].values)
    for col in _LOOKUP_COLS:
        chunk[f"s1_{col}"] = s1_fields[col].values

    cand_ids = chunk["candidate_entity_id"]
    is_s2 = cand_ids.str.startswith("S2-")
    is_s3 = cand_ids.str.startswith("S3-")

    for col in _LOOKUP_COLS:
        vals = pd.Series(index=chunk.index, dtype="object")
        vals.loc[is_s2] = s2_df.reindex(cand_ids[is_s2].values)[col].values
        vals.loc[is_s3] = s3_df.reindex(cand_ids[is_s3].values)[col].values
        chunk[f"cand_{col}"] = vals

    return chunk


def _compute_features(chunk):
    a_name = chunk["s1_name_norm"].fillna("").tolist()
    b_name = chunk["cand_name_norm"].fillna("").tolist()
    a_addr = chunk["s1_addr_norm"].fillna("").tolist()
    b_addr = chunk["cand_addr_norm"].fillna("").tolist()
    a_name_c = chunk["s1_name_compact"].fillna("").tolist()
    b_name_c = chunk["cand_name_compact"].fillna("").tolist()
    a_country = chunk["s1_country"].fillna("").str.lower().tolist()
    b_country = chunk["cand_country"].fillna("").str.lower().tolist()

    rows = [
        sim.pairwise_features(
            a_name[i], b_name[i], a_addr[i], b_addr[i],
            a_name_c[i], b_name_c[i], a_country[i], b_country[i],
        )
        for i in range(len(chunk))
    ]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)

def build_features(candidate_pairs_path, s1_path, s2_path, s3_path, output_path=None, gt_path=None, chunk_size=200_000):
    """Pairwise features for every candidate pair.

    Memory-safe: when `output_path` is given, each chunk is appended to disk
    and nothing accumulates in RAM (returns None). Only when `output_path`
    is None is the full table concat'ed in memory and returned.
    """
    s1_df = _load_source_lookup(s1_path, "S1")
    s2_df = _load_source_lookup(s2_path, "S2")
    s3_df = _load_source_lookup(s3_path, "S3")

    pairs = _explode_candidates(candidate_pairs_path)
    print(f"total candidate pairs: {len(pairs):,}", flush=True)

    gt_pairs = _load_ground_truth_pairs(gt_path) if gt_path else None

    streaming = output_path is not None
    if streaming:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()  # fresh reproducible run
    out_chunks, first_write = [] if not streaming else None, True
    total_pairs, total_pos = 0, 0
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs.iloc[start:start + chunk_size].reset_index(drop=True).copy()
        chunk = _attach_fields(chunk, s1_df, s2_df, s3_df)
        feats = _compute_features(chunk)
        result_chunk = pd.concat(
            [chunk[["source1_entity_id", "candidate_entity_id"]], feats], axis=1
        )
        if gt_pairs is not None:
            result_chunk["label"] = [
                1 if (s1, c) in gt_pairs else 0
                for s1, c in zip(result_chunk["source1_entity_id"], result_chunk["candidate_entity_id"])
            ]
            total_pos += int(result_chunk["label"].sum())
        total_pairs += len(result_chunk)
        mb = result_chunk.memory_usage(deep=True, index=False).sum() / 1e6
        if streaming:
            result_chunk.to_csv(output_path, sep="\t", index=False,
                                mode="w" if first_write else "a", header=first_write)
            first_write = False
        else:
            out_chunks.append(result_chunk)
        print(f"features: processed {min(start + chunk_size, len(pairs)):,}/{len(pairs):,} "
              f"(chunk ~{mb:,.1f}MB)", flush=True)
        del chunk, feats, result_chunk
        gc.collect()

    if gt_pairs is not None:
        print(f"positive rate in feature table: {total_pos / total_pairs:.4%}", flush=True)

    if streaming:
        print(f"DONE -> {output_path} rows={total_pairs:,}", flush=True)
        return None

    result = pd.concat(out_chunks, ignore_index=True)
    del out_chunks
    gc.collect()
    return result

def build_train_features(candidate_pairs_path, output_path=None, chunk_size=200_000):
    return build_features(
        candidate_pairs_path, dl.TRAIN_S1, dl.TRAIN_S2, dl.TRAIN_S3,
        output_path=output_path, gt_path=dl.TRAIN_GT, chunk_size=chunk_size,
    )