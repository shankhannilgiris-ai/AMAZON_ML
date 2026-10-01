"""Stages 4-6: pruner (cheap ranker), main LightGBM ranker, scoring."""
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from .features import cheap_features, heavy_features, group_features
from .retrieve import FAMILIES

PRUNER_PATH = C.MODELS / "pruner.txt"
RANKER_PATH = C.MODELS / "ranker.txt"

RET_FEATS = ["ret_score", "ret_rank"] + ["k_" + f for f in FAMILIES]
CHEAP = ["c_n_ratio", "c_n_tset", "c_a_tset"]


def _gap_rank(df, cols):
    g = df.groupby("q", sort=False)
    for c in cols:
        df[c + "_g"] = (df[c] - g[c].transform("max")).astype(np.float32)
    df["n_cand0"] = g.q.transform("size").astype(np.float32)
    return df


PRUNE_FEATS = RET_FEATS + CHEAP + [c + "_g" for c in ["ret_score"] + CHEAP] + [
    "n_cand0", "q_addr_empty", "q_native", "country"]


def prune_frame(cand, Q, S):
    q, s1 = cand.q.values, cand.s1.values
    f = cheap_features(Q.name_n.values[q], S.name_n.values[s1], Q.addr_n.values[q], S.addr_n.values[s1])
    for k, v in f.items():
        cand[k] = v
    cand["q_addr_empty"] = (Q.addr_len.values[q] == 0).astype(np.float32)
    cand["q_native"] = Q.native.values[q].astype(np.float32)
    cand["country"] = Q.country.values[q].astype(np.float32)
    return _gap_rank(cand, ["ret_score"] + CHEAP)


LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, num_threads=C.N_JOBS, verbose=-1, seed=C.SEED)


def train_lgb(X, y, Xv, yv, path, rounds=3000, params=None, names=None):
    p = dict(LGB_PARAMS, **(params or {}))
    dtr = lgb.Dataset(X, y, feature_name=names or "auto", free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    m = lgb.train(p, dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    m.save_model(str(path))
    return m


def top_per_query(df, score_col, k):
    df = df.sort_values(["q", score_col], ascending=[True, False], kind="stable")
    r = df.groupby("q", sort=False).cumcount()
    return df[r.values < k]


def heavy_frame(part, Q, S, pool):
    f = heavy_features(part.q.values, part.s1.values, Q, S, pool)
    for k, v in f.items():
        part[k] = v
    return group_features(part)


def ranker_features(df):
    drop = {"q", "s1", "label", "src", "is_val"}
    return [c for c in df.columns if c not in drop]
