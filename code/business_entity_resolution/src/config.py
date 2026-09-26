"""Paths and tunables for the entity resolution pipeline.

Every path can be overridden with an environment variable so the package runs
unchanged on another machine (see README.md).
"""
import os
import zlib
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get(
    "ER_DATA",
    "/home/arin/Documents/dataset/6ab10eb3b23ba_student_resource/student_resource/dataset",
))
WORK = Path(os.environ.get("ER_WORK", PROJECT / "work"))
OUTPUT = Path(os.environ.get("ER_OUTPUT", PROJECT / "output"))

WORKERS = int(os.environ.get("ER_WORKERS", "4"))
CHUNK = int(os.environ.get("ER_CHUNK", "100000"))

# ---- blocking -------------------------------------------------------------
# A key shared by more S1 records than its cap is too generic to block on.
CAP_NAME_SINGLE = 30        # single-name-token keys (NS) for queries with an address
CAP = 50                    # every other key family for queries with an address
CAP_NO_ADDR = 200           # all name keys for queries without an address

# How many of the rarest tokens feed each key family. S1 emits from more tokens
# than the query side so a query that lost or misspelled one token still collides.
TOP = {
    "s1": dict(np_=4, an=4, ap=4, na_n=2, na_a=3, nn=3),
    "q": dict(np_=3, an=2, ap=3, na_n=2, na_a=2, nn=2),
}
MAX_NUMS = 3                # address number tokens used per record

# NS name token | NP name token pair | AN address number + address word | AP address word pair
# NA name token + address word | NN address number + name token
KEY_TYPES = ["NS", "NP", "AN", "AP", "NA", "NN"]
KEY_WEIGHTS = [1.0, 1.5, 1.5, 1.0, 1.2, 1.5]

# Candidate score = sum over shared keys of weight / log2(1 + key frequency)
#                   + W_NAME * Dice overlap of the two name-token sets.
# The overlap term separates "Sunrise Industries" from "Sunrise Industries Engineering"
# when both share exactly the same keys with the query.
W_NAME = float(os.environ.get("ER_W_NAME", "1.0"))
W_NAME_TRY = (0, 0.5, 1, 2, 4)  # train only: true-pair rank under each weight, for tuning

K_STORE = 10                # candidates stored per query record (before the final policy)
K_ADDR = 5                  # final policy: top-K per query record with an address
K_NO_ADDR = 10              # final policy: top-K per query record without an address
HUB_CAP = 200               # final policy: max candidates kept per S1 entity

# ---- typo correction --------------------------------------------------------
TYPO_MIN_LEN = 4


def typo_max_dist(n: int) -> int:
    return 1 if n < 6 else 2


def s1_path(split: str) -> Path:
    return DATA / split / f"{split}_source1.tsv"


def query_path(split: str, src: int) -> Path:
    return DATA / split / f"{split}_source{src}.tsv"


def gt_path() -> Path:
    return DATA / "train" / "train_ground_truth.tsv"


def is_val(s1_entity_id: str) -> bool:
    """Deterministic 80/20 train/validation split over Source 1 entities."""
    return zlib.crc32(s1_entity_id.encode()) % 10 >= 8
