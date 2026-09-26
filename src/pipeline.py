import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.candidate_generator import generate_train_candidate_pairs
from src.features import build_train_features
from src import matcher

OUTPUT_DIR = REPO_ROOT / "output"
TRAIN_CANDIDATES_PATH = OUTPUT_DIR / "candidate_pairs_train.tsv"
TRAIN_FEATURES_PATH = OUTPUT_DIR / "train_features.tsv"
MODEL_PATH = OUTPUT_DIR / "models" / "lgbm_matcher.txt"
THRESHOLD_PATH = OUTPUT_DIR / "models" / "threshold.json"

BLOCKING_KWARGS = dict(batch_size=25000, max_block_size=5000, max_per_s1=200)

def main():
    print(f"repo root : {REPO_ROOT}", flush=True)
    print(f"output dir: {OUTPUT_DIR} (outside src/)", flush=True)
    print("=== [1/3] blocking: train candidates ===", flush=True)
    generate_train_candidate_pairs(TRAIN_CANDIDATES_PATH, **BLOCKING_KWARGS)

    print("=== [2/3] features: train ===", flush=True)
    build_train_features(TRAIN_CANDIDATES_PATH, output_path=TRAIN_FEATURES_PATH)

    print("=== [3/3] train + tune threshold ===", flush=True)
    model, threshold, val_f05 = matcher.train_and_tune(
        TRAIN_FEATURES_PATH, model_out=MODEL_PATH, threshold_out=THRESHOLD_PATH
    )
    print(f"val macro F0.5 = {val_f05:.4f} @ threshold={threshold:.2f}", flush=True)

    print("DONE (train-only; test stages added later).", flush=True)
    print(f"  {TRAIN_CANDIDATES_PATH}", flush=True)
    print(f"  {TRAIN_FEATURES_PATH}", flush=True)
    print(f"  {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
