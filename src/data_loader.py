from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Layout-aware dataset root: local checkout nests student_resource under
# datasets/, while on Colab the repo root typically sits directly above
# student_resource/. First existing layout wins, so moving to Colab needs
# no code change as long as one of these exists.
_CANDIDATE_BASES = [
    REPO_ROOT / "datasets" / "student_resource" / "student_resource" / "dataset",
    REPO_ROOT / "student_resource" / "dataset",
]
BASE = next((p for p in _CANDIDATE_BASES if (p / "train").is_dir()), _CANDIDATE_BASES[0])

TRAIN_S1 = BASE / "train" / "train_source1.tsv"
TRAIN_S2 = BASE / "train" / "train_source2.tsv"
TRAIN_S3 = BASE / "train" / "train_source3.tsv"
TRAIN_GT = BASE / "train" / "train_ground_truth.tsv"

TEST_S1 = BASE / "test" / "test_source1.tsv"
TEST_S2 = BASE / "test" / "test_source2.tsv"
TEST_S3 = BASE / "test" / "test_source3.tsv"

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]


def _read_tsv(path, **kwargs):
    kwargs.setdefault("sep", "\t")
    kwargs.setdefault("dtype", "string")
    return pd.read_csv(path, **kwargs)


def load_source(path, usecols=None):
    """Load one source TSV (S1/S2/S3, train or test)."""
    return _read_tsv(path, keep_default_na=True, usecols=usecols)


def load_train_s1(usecols=None):
    return load_source(TRAIN_S1, usecols=usecols)


def load_train_s2(usecols=None):
    return load_source(TRAIN_S2, usecols=usecols)


def load_train_s3(usecols=None):
    return load_source(TRAIN_S3, usecols=usecols)


def load_test_s1(usecols=None):
    return load_source(TEST_S1, usecols=usecols)


def load_test_s2(usecols=None):
    return load_source(TEST_S2, usecols=usecols)


def load_test_s3(usecols=None):
    return load_source(TEST_S3, usecols=usecols)


def load_train_ground_truth():
    return _read_tsv(TRAIN_GT, keep_default_na=False)


def load_train_s1_and_gt():
    """Mirror of notebook cell 1: full S1 + full GT."""
    s1 = load_train_s1()
    gt = load_train_ground_truth()
    return s1, gt
