"""Ground truth -> integer label arrays + grouped train/val split.

label_s{2,3}[q] = index of the true S1 record for query q, or -1 (distractor).
Validation is grouped by S1 cluster: a held-out S1 id and all its matches
are never seen by any trained model.
"""
import numpy as np
import pandas as pd

from . import config as C


def load_labels(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame):
    dst = C.CACHE / "train_labels.npz"
    if dst.exists():
        z = np.load(dst)
        return z["y2"], z["y3"], z["val_s1"]
    gt = pd.read_csv(C.RAW / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    pairs = pairs[pairs.m != ""]
    s1_pos = pd.Series(np.arange(len(s1), dtype=np.int32), index=s1.eid.values)
    p1 = s1_pos.reindex(pairs.source1_entity_id.values).values
    out = []
    for df in (s2, s3):
        y = np.full(len(df), -1, dtype=np.int32)
        pos = pd.Series(np.arange(len(df), dtype=np.int32), index=df.eid.values)
        q = pos.reindex(pairs.m.values).values
        ok = ~np.isnan(q)
        y[q[ok].astype(np.int64)] = p1[ok]
        out.append(y)
    rng = np.random.default_rng(C.SEED)
    val_s1 = rng.random(len(s1)) < C.VAL_FRAC
    np.savez(dst, y2=out[0], y3=out[1], val_s1=val_s1)
    return out[0], out[1], val_s1


def query_is_val(y: np.ndarray, val_s1: np.ndarray, seed_offset: int) -> np.ndarray:
    """Matched queries follow their S1 cluster; distractors are split randomly."""
    rng = np.random.default_rng(C.SEED + seed_offset)
    isval = rng.random(len(y)) < C.VAL_FRAC
    m = y >= 0
    isval[m] = val_s1[y[m]]
    return isval


def orphan_view(s1, y2, y3, val):
    """Drop ORPHAN_FRAC of S1 so train matches test's orphan rate.
    Returns (s1_kept, y2', y3', val') with S1 indices renumbered; queries of
    dropped entities become distractors (-1)."""
    rng = np.random.default_rng(C.SEED + 99)
    drop = rng.random(len(s1)) < C.ORPHAN_FRAC
    newpos = np.cumsum(~drop) - 1
    out = []
    for y in (y2, y3):
        z = np.full(len(y), -1, dtype=np.int32)
        ok = (y >= 0)
        ok[ok] = ~drop[y[ok]]
        z[ok] = newpos[y[ok]]
        out.append(z)
    s1k = s1[~drop].reset_index(drop=True)
    if "dup" in s1k.columns and "compact" in s1k.columns:
        key = s1k.country.astype(str) + "|" + s1k.compact
        s1k["dup"] = key.map(key.value_counts()).astype(np.int32).values
    return s1k, out[0], out[1], val[~drop]
