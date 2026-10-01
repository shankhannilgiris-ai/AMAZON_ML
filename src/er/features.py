"""Stage 3 - pair features.

cheap_features: run on every retrieved pair (~300M); rapidfuzz cpdist only.
heavy_features: run on pruned pairs (~80M); full name/address/number set.
group_features: query-relative features (f - max_f within the query's
candidate list), which is what an argmax assignment actually needs.

All rapidfuzz calls are element-wise `cpdist(..., workers=-1)` (C++, all
cores). Python-level token logic runs in a process pool.
"""
import math
from multiprocessing import Pool
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from . import config as C
from .text import core_tokens, skeleton, ADDR_STOP

F32 = np.float32


def _cp(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=F32)


def cheap_features(qn, sn, qa, sa):
    return {
        "c_n_ratio": _cp(qn, sn, fuzz.ratio),
        "c_n_tset": _cp(qn, sn, fuzz.token_set_ratio),
        "c_a_tset": _cp(qa, sa, fuzz.token_set_ratio),
    }


# ------------------------------------------------------------------ python-level
_STOP = _IDF = None


def _init(stop, idf):
    global _STOP, _IDF
    _STOP, _IDF = stop, idf


def _num_list(s):
    return [x for x in s.split() if x]


def _py_block(args):
    qn, sn, qa, sa, qnum, snum, qst, sst = args
    n = len(qn)
    out = np.zeros((n, 16), dtype=F32)
    idf = _IDF; default_idf = 12.0
    for i in range(n):
        qt = set(qn[i].split()); st = set(sn[i].split())
        inter = qt & st; uni = qt | st
        out[i, 0] = len(inter) / len(uni) if uni else 0
        out[i, 1] = len(inter) / len(st) if st else 0          # containment of S1 name in query
        out[i, 2] = len(inter) / len(qt) if qt else 0
        wi = sum(idf.get(t, default_idf) for t in inter)
        wu = sum(idf.get(t, default_idf) for t in uni)
        out[i, 3] = wi / wu if wu else 0                        # idf-weighted jaccard
        out[i, 4] = max((idf.get(t, default_idf) for t in (uni - inter)), default=0)  # rarest unmatched token
        qc = core_tokens(qn[i], _STOP); sc = core_tokens(sn[i], _STOP)
        qcs, scs = set(qc), set(sc)
        out[i, 5] = len(qcs & scs) / len(qcs | scs) if (qcs | scs) else 0
        out[i, 6] = float(qc[:1] == sc[:1])
        # numbers
        a, b = _num_list(qnum[i]), _num_list(snum[i])
        out[i, 7] = len(a) == 0
        if a and b:
            sa_, sb_ = set(a), set(b)
            out[i, 8] = len(sa_ & sb_) / len(sa_ | sb_)
            out[i, 9] = float(a[0] == b[0])
            out[i, 10] = float(a[0] in sb_)
            out[i, 11] = math.log1p(abs(int(a[0][:9]) - int(b[0][:9])))
            out[i, 12] = float("".join(sorted(a)) == "".join(sorted(b)))
            out[i, 13] = len(sa_ - sb_)
        else:
            out[i, 8:14] = -1
        # address alpha tokens (street + locality)
        at = {t for t in qa[i].replace(",", " ").split() if not t.isdigit() and t not in ADDR_STOP}
        bt = {t for t in sa[i].replace(",", " ").split() if not t.isdigit() and t not in ADDR_STOP}
        out[i, 14] = len(at & bt) / len(at) if at else -1
        out[i, 15] = -1 if not (qst[i] and sst[i]) else float(qst[i] == sst[i])
    return out


PY_NAMES = ["n_jac", "n_cont_s", "n_cont_q", "n_idf_jac", "n_max_unmatched_idf", "core_jac",
            "core_first_eq", "q_no_nums", "num_jac", "num_first_eq", "num_first_in", "num_first_logdiff",
            "num_set_eq", "num_q_extra", "addr_tok_cov", "state_eq"]


def heavy_features(q, s1, Q, S, pool, block=200_000):
    qn = Q.name_n.values[q]; sn = S.name_n.values[s1]
    qa = Q.addr_n.values[q]; sa = S.addr_n.values[s1]
    f = {}
    f["n_ratio"] = _cp(qn, sn, fuzz.ratio)
    f["n_pratio"] = _cp(qn, sn, fuzz.partial_ratio)
    f["n_tsort"] = _cp(qn, sn, fuzz.token_sort_ratio)
    f["n_tset"] = _cp(qn, sn, fuzz.token_set_ratio)
    f["n_jw"] = _cp(qn, sn, JaroWinkler.normalized_similarity)
    f["n_lev"] = _cp(qn, sn, Levenshtein.normalized_similarity)
    qc = Q.compact.values[q]; sc = S.compact.values[s1]
    f["cp_ratio"] = _cp(qc, sc, fuzz.ratio)
    f["cp_pratio"] = _cp(qc, sc, fuzz.partial_ratio)
    f["cp_jw"] = _cp(qc, sc, JaroWinkler.normalized_similarity)
    f["cp_eq"] = (qc == sc).astype(F32)
    f["sk_ratio"] = _cp(Q.skel.values[q], S.skel.values[s1], fuzz.ratio)
    f["nu_tset"] = _cp(Q.name_u.values[q], sn, fuzz.token_set_ratio)   # pre-transliteration
    f["a_ratio"] = _cp(qa, sa, fuzz.ratio)
    f["a_pratio"] = _cp(qa, sa, fuzz.partial_ratio)
    f["a_tsort"] = _cp(qa, sa, fuzz.token_sort_ratio)
    f["a_tset"] = _cp(qa, sa, fuzz.token_set_ratio)
    f["a_first_ratio"] = _cp(Q.addr_first.values[q], S.addr_first.values[s1], fuzz.ratio)
    f["a_last_in"] = _cp(S.addr_last.values[s1], qa, fuzz.partial_ratio)  # S1 locality inside query addr
    f["q_addr_len"] = Q.addr_len.values[q].astype(F32)
    f["s_addr_len"] = S.addr_len.values[s1].astype(F32)
    f["q_name_len"] = Q.name_len.values[q].astype(F32)
    f["s_name_len"] = S.name_len.values[s1].astype(F32)
    f["q_dom"] = Q.dom.values[q].astype(F32)
    f["q_native"] = Q.native.values[q].astype(F32)
    f["country"] = Q.country.values[q].astype(F32)
    f["s_dup"] = S.dup.values[s1].astype(F32)                 # how many S1 share this core name
    f["s_addr_empty"] = (S.addr_len.values[s1] == 0).astype(F32)
    qnum = Q.nums.values[q]; snum = S.nums.values[s1]
    qst = Q.state.values[q]; sst = S.state.values[s1]
    args = [(qn[i:i + block], sn[i:i + block], qa[i:i + block], sa[i:i + block],
             qnum[i:i + block], snum[i:i + block], qst[i:i + block], sst[i:i + block])
            for i in range(0, len(q), block)]
    py = np.concatenate(list(pool.imap(_py_block, args))) if args else np.zeros((0, 16), F32)
    for j, nme in enumerate(PY_NAMES):
        f[nme] = py[:, j]
    return f


def add_derived_columns(df, stop):
    """Per-record helper columns used by heavy_features (computed once)."""
    toks = [core_tokens(n, stop) for n in df.name_n.values]
    df["compact"] = ["".join(t) for t in toks]
    df["skel"] = [" ".join(skeleton(x) for x in t) for t in toks]
    comps = [a.split(", ") if a else [""] for a in df.addr_n.values]
    df["addr_first"] = [c[0] for c in comps]
    df["addr_last"] = [c[-1] for c in comps]
    df["addr_len"] = df.addr_n.str.len().astype(np.int16)
    df["name_len"] = df.name_n.str.len().astype(np.int16)
    return df


GROUP_COLS = ["prune_score", "ret_score", "n_tset", "n_ratio", "cp_ratio", "a_tset", "a_ratio",
              "num_jac", "n_idf_jac", "addr_tok_cov"]


def group_features(df):
    """Query-relative features. Candidates of a query are contiguous."""
    g = df.groupby("q", sort=False)
    df["n_cand"] = g.q.transform("size").astype(F32)
    for c in GROUP_COLS:
        mx = g[c].transform("max")
        df[f"{c}_gap"] = (df[c] - mx).astype(F32)
        # second best, to measure how contested the query is
        df[f"{c}_rank"] = g[c].rank(ascending=False, method="min").astype(F32)
    return df


def token_idf(s1):
    from collections import Counter
    cnt = Counter(t for n in s1.name_n.values for t in set(n.split()))
    N = len(s1)
    return {t: float(math.log(N / c)) for t, c in cnt.items()}
