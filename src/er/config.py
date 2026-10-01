from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT                      # *_processed.tsv live in the project root
CACHE = ROOT / "data" / "cache"
MODELS = ROOT / "models"
OUT = ROOT / "output"
for p in (CACHE, MODELS, OUT):
    p.mkdir(parents=True, exist_ok=True)

N_JOBS = max(1, (os.cpu_count() or 4) - 1)
SOURCES = (1, 2, 3)
COUNTRY_CODE = {"US": 0, "India": 1}

# ---- retrieval
KEY_DF_CAP = 400          # drop blocking keys shared by more S1 records than this
TOPK_RETRIEVE = 30        # candidates per query kept after sparse retrieval
TOPK_RANK = 8             # candidates per query kept for the heavy ranker
QUERY_CHUNK = 80_000      # queries per retrieval chunk (bounds peak RAM: ~60M postings)

# ---- validation: fraction of S1 clusters held out (grouped by S1 id)
VAL_FRAC = 0.15
# ---- test has ~2.82 S2 and 2.93 S3 records per S1 vs 2.28/2.40 in train: the
# test S1 file is missing entities whose S2/S3 records remain as orphans.
# Dropping this fraction of train S1 reproduces that ratio.
ORPHAN_FRAC = 0.0        # submitted model (v1) was trained without orphan simulation; 0.19 = v2 experiment
SEED = 42


def raw_path(split: str, src: int) -> Path:
    return RAW / f"{split}_source{src}_processed.tsv"


def cache_path(split: str, src: int) -> Path:
    return CACHE / f"{split}_s{src}.parquet"
