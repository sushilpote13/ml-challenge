import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

from src.features import FEATURE_NAMES

TRAIN_FEATURES_PATH = "output/train_features.tsv"
MODEL_OUT_PATH = "output/models/lgbm_matcher.txt"
THRESHOLD_OUT_PATH = "output/models/threshold.json"
VAL_FRACTION = 0.2
RANDOM_SEED = 42
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 30,
    "verbosity": -1,
    "seed": RANDOM_SEED,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50


def _detect_device():
    try:
        test_params = dict(LGBM_PARAMS)
        test_params["device_type"] = "cuda"
        ds = lgb.Dataset(np.zeros((10, 2), dtype=np.float32), label=np.zeros(10))
        lgb.train(test_params, ds, num_boost_round=1)
        print("[matcher] LightGBM CUDA device available, using device_type=cuda", flush=True)
        return "cuda"
    except Exception:
        print("[matcher] LightGBM CUDA not available/usable, using CPU", flush=True)
        return "cpu"

def load_feature_table(path):
    dtype = {name: "float32" for name in FEATURE_NAMES}
    dtype.update({"source1_entity_id": "string", "candidate_entity_id": "string", "label": "int8"})
    df = pd.read_csv(path, sep="\t", dtype=dtype)
    mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"[matcher] loaded features: rows={len(df):,} ~{mb:,.1f}MB "
          f"pos_rate={df['label'].mean():.4%}", flush=True)
    return df


def group_split(df, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df["source1_entity_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(f"[matcher] split: train_pairs={len(train_df):,} "
          f"({train_df['source1_entity_id'].nunique():,} s1 entities), "
          f"val_pairs={len(val_df):,} "
          f"({val_df['source1_entity_id'].nunique():,} s1 entities)", flush=True)
    return train_df, val_df

def train_model(train_df, val_df, params=None):
    params = dict(params or LGBM_PARAMS)
    pos = int(train_df["label"].sum())
    neg = len(train_df) - pos
    # Guard: degenerate single-class splits (tiny debug subsets) would set
    # scale_pos_weight <= 0 and crash LightGBM. No-op on real data (both > 0).
    params["scale_pos_weight"] = (neg / pos) if (pos > 0 and neg > 0) else 1.0
    print(f"[matcher] scale_pos_weight={params['scale_pos_weight']:.2f} "
          f"(pos={pos:,} neg={neg:,})", flush=True)

    device = _detect_device()
    if device == "cuda":
        params["device_type"] = "cuda"

    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df["label"].values
    X_val = val_df[FEATURE_NAMES].values
    y_val = val_df["label"].values

    ds_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    ds_val = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=ds_train)

    model = lgb.train(
        params,
        ds_train,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_train, ds_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    del X_train, y_train, ds_train
    gc.collect()
    return model

def f_beta_for_entity(pred_set, true_set, beta=0.5):
    if not pred_set and not true_set:
        return 1.0
    if not pred_set:          # true_set non-empty, nothing predicted
        return 0.0
    if not true_set:          # predicted something, but no true matches
        return 0.0
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f05(pred_by_s1, true_by_s1, all_s1_ids):
    scores = [
        f_beta_for_entity(pred_by_s1.get(s1, set()), true_by_s1.get(s1, set()))
        for s1 in all_s1_ids
    ]
    return float(np.mean(scores))


def _true_by_s1_from_val(val_df):
    truth = {}
    for s1, cand, label in zip(val_df["source1_entity_id"], val_df["candidate_entity_id"], val_df["label"]):
        if label == 1:
            truth.setdefault(s1, set()).add(cand)
    return truth


def tune_threshold(model, val_df, thresholds=THRESHOLD_GRID):
    X_val = val_df[FEATURE_NAMES].values
    scores = model.predict(X_val, num_iteration=model.best_iteration)
    s1_ids = val_df["source1_entity_id"].values
    cand_ids = val_df["candidate_entity_id"].values
    true_by_s1 = _true_by_s1_from_val(val_df)
    all_val_s1 = val_df["source1_entity_id"].unique().tolist()

    best_t, best_f05 = 0.5, -1.0
    for t in thresholds:
        keep = scores >= t
        pred_by_s1 = {}
        for s1, cand in zip(s1_ids[keep], cand_ids[keep]):
            pred_by_s1.setdefault(s1, set()).add(cand)
        f05 = macro_f05(pred_by_s1, true_by_s1, all_val_s1)
        if f05 > best_f05:
            best_f05, best_t = f05, float(t)

    print(f"[matcher] best threshold={best_t:.2f} val macro F0.5={best_f05:.4f}", flush=True)
    return best_t, best_f05

def predict_matching_results(model, threshold, features_path, all_s1_ids,
                              output_path, chunk_size=500_000):
    dtype = {name: "float32" for name in FEATURE_NAMES}
    dtype.update({"source1_entity_id": "string", "candidate_entity_id": "string"})
    matched = {}
    reader = pd.read_csv(features_path, sep="\t", dtype=dtype,
                          usecols=["source1_entity_id", "candidate_entity_id", *FEATURE_NAMES],
                          chunksize=chunk_size)
    for chunk in reader:
        scores = model.predict(chunk[FEATURE_NAMES].values, num_iteration=model.best_iteration)
        keep = scores >= threshold
        for s1, cand in zip(chunk["source1_entity_id"].values[keep],
                            chunk["candidate_entity_id"].values[keep]):
            matched.setdefault(s1, []).append(cand)
        del chunk, scores
        gc.collect()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in all_s1_ids:
            f.write(f"{s1}\t{','.join(matched.get(s1, []))}\n")
    print(f"[matcher] wrote matching_results -> {output_path} "
          f"rows={len(all_s1_ids):,}", flush=True)


def load_all_s1_ids(s1_path):
    return pd.read_csv(s1_path, sep="\t", dtype="string", usecols=["entity_id"])["entity_id"].tolist()

def train_and_tune(features_path, model_out=MODEL_OUT_PATH, threshold_out=THRESHOLD_OUT_PATH):
    df = load_feature_table(features_path)
    train_df, val_df = group_split(df)
    del df
    gc.collect()

    model = train_model(train_df, val_df)
    threshold, val_f05 = tune_threshold(model, val_df)

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_out))
    with open(threshold_out, "w") as f:
        json.dump({"threshold": threshold, "val_macro_f05": val_f05}, f, indent=2)
    print(f"[matcher] saved model -> {model_out}, threshold -> {threshold_out}", flush=True)

    importances = sorted(zip(FEATURE_NAMES, model.feature_importance(importance_type="gain")),
                         key=lambda x: -x[1])
    print("[matcher] top feature importances (gain):", flush=True)
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:,.0f}", flush=True)

    return model, threshold, val_f05

def main():
    train_and_tune(TRAIN_FEATURES_PATH)

if __name__ == "__main__":
    main()