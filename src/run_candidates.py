# from candidate_generator import generate_train_candidate_pairs
from src.candidate_generator import generate_train_candidate_pairs
OUTPUT = "output/candidate_pairs_train.tsv"
BATCH_SIZE = 25000
MAX_BLOCK_SIZE = 5000
MAX_PER_S1 = 200
LIMIT_S1 = None

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
