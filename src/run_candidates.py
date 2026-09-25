"""Runner for Phase 3 candidate generation (TRAIN data only).

Usage (from repo root):
    python src/run_candidates.py

Edit SETTINGS below, then run. No command-line flags: the library
functions in src/candidate_generator.py do the work; this script only
wires the train trio -> output file. Test data stays untouched.
"""

from src.candidate_generator import generate_train_candidate_pairs

# ---------------- SETTINGS (edit me) ----------------
OUTPUT = "output/candidate_pairs_train.tsv"
BATCH_SIZE = 25000                    # S1 rows per batch (25k-50k)
MAX_BLOCK_SIZE = 5000                 # drop blocks bigger than this (recall-first cap)
MAX_PER_S1 = 200                      # safety cap on candidates per S1
LIMIT_S1 = None                       # None = full train run; e.g. 500 for a smoke test
# ----------------------------------------------------


def main():
    generate_train_candidate_pairs(
        OUTPUT,
        batch_size=BATCH_SIZE,
        max_block_size=MAX_BLOCK_SIZE,
        max_per_s1=MAX_PER_S1,
        limit_s1=LIMIT_S1,
    )
    print(f"DONE -> {OUTPUT}")


if __name__ == "__main__":
    main()
