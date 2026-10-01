"""Stage 2 - multi-pass inverted-index blocking + candidate generation.

Problem framing (verified on train GT): every S2/S3 record belongs to AT MOST
ONE S1 entity. So we retrieve, for every S2/S3 record (the "query"), a short
list of S1 candidates. That turns an O(N1*N2) pair space into O(N2*K).

Each record emits hashed blocking keys from several families; the S1 side is
an inverted index (sorted uint64 keys -> postings). A candidate's retrieval
score is the IDF-weighted sum of shared keys, split per family so the ranker
can learn which families matter. Keys whose S1 document frequency exceeds
KEY_DF_CAP are dropped (the 'MemoryError' blocks of the naive approach).

Peak memory is bounded by QUERY_CHUNK, not by dataset size.
"""
from multiprocessing import Pool
import itertools
import numpy as np
import pandas as pd

from . import config as C
from .text import core_tokens, skeleton, addr_tokens

FAMILIES = ["compact", "prefix", "tok", "tokpair", "skel", "skelpair",
            "numstreet", "addrpair", "tok_num", "addr_exact",
            "name_addr", "pre_addr", "suf_addr", "skel_addr"]
FAM_W = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 1.0, 1.5,
                  1.0, 0.6, 0.6, 1.0], dtype=np.float32)
NF = len(FAMILIES)

_STOP = _STOP_SKEL = None


def _init(stop, stop_skel):
    global _STOP, _STOP_SKEL
    _STOP, _STOP_SKEL = stop, stop_skel


def _record_keys(country, name_n, addr_n, nums, index_side):
    ks = []
    c = str(country)
    toks = core_tokens(name_n, _STOP)
    compact = "".join(toks)
    if compact:
        ks.append((0, compact))
        if len(compact) >= 8:
            ks.append((1, compact[:6]))
    toks_u = list(dict.fromkeys(t for t in toks if len(t) >= 2))
    for t in toks_u[:6]:
        ks.append((2, t))
    for a, b in itertools.combinations(sorted(toks_u[:4]), 2):
        ks.append((3, a + "|" + b))
    if country == 1:     # India: phonetic skeleton space absorbs transliteration
        sk = [skeleton(t) for t in name_n.split()]
        sk = list(dict.fromkeys(s for s in sk if len(s) >= 2 and s not in _STOP_SKEL))
        for s in sk[:6]:
            ks.append((4, s))
        for a, b in itertools.combinations(sorted(sk[:4]), 2):
            ks.append((5, a + "|" + b))
    at = [t for t in addr_tokens(addr_n) if not t.isdigit() and len(t) >= 3]
    at = list(dict.fromkeys(at))[:8]
    nl = [n for n in nums.split() if len(n) <= 6][:3]
    for j, n in enumerate(nl):
        variants = [n]
        if index_side and j == 0 and len(n) >= 2:     # absorb +-1 house-number noise
            v = int(n)
            variants += [str(v - 1), str(v + 1)]
        for v in variants:
            for t in at:
                ks.append((6, v + "|" + t))
            for t in toks_u[:3]:
                ks.append((8, v + "|" + t))
    at = at[:5]
    for a, b in itertools.combinations(sorted(at), 2):
        ks.append((7, a + "|" + b))
    if addr_n:
        ks.append((9, " ".join(sorted(addr_tokens(addr_n)))))
    # name x address combinations: 54% of S1 names are shared by several S1
    # entities, so the discriminative unit is (name, any address token).
    # prefix/suffix variants survive single-typo names ('herltah').
    if addr_n and compact:
        aa = list(dict.fromkeys(t for t in addr_tokens(addr_n)
                                if (len(t) >= 3 or t.isdigit()) and len(t) <= 12))[:10]
        pre, suf = compact[:5], compact[-5:]
        for t in aa:
            ks.append((10, compact + "|" + t))
            if len(compact) >= 8:
                ks.append((11, pre + "|" + t))
                ks.append((12, suf + "|" + t))
        if country == 1:
            skc = "".join(s for s in (skeleton(t) for t in toks) if s not in _STOP_SKEL)
            if skc:
                for t in aa:
                    ks.append((13, skc + "|" + (t if t.isdigit() else skeleton(t))))
    seen = set()
    out = []
    for f, k in ks:
        s = f"{c}{f}{k}"
        if s not in seen:
            seen.add(s); out.append((f, s))
    return out


def _chunk_keys(args):
    start, country, name_n, addr_n, nums, index_side = args
    rows, fams, strs = [], [], []
    for i in range(len(country)):
        for f, s in _record_keys(country[i], name_n[i], addr_n[i], nums[i], index_side):
            rows.append(start + i); fams.append(f); strs.append(s)
    keys = pd.util.hash_array(np.array(strs, dtype=object))   # deterministic across processes
    return (np.array(rows, dtype=np.int32), np.array(fams, dtype=np.int8), keys)


def emit_keys(df, pool, index_side, n=25_000):
    args = [(i, df.country.values[i:i + n], df.name_n.values[i:i + n],
             df.addr_n.values[i:i + n], df.nums.values[i:i + n], index_side)
            for i in range(0, len(df), n)]
    for r in pool.imap(_chunk_keys, args):
        yield r


class S1Index:
    def __init__(self, s1, pool):
        parts = list(emit_keys(s1, pool, index_side=True))
        rows = np.concatenate([p[0] for p in parts]); fams = np.concatenate([p[1] for p in parts])
        keys = np.concatenate([p[2] for p in parts]); del parts
        order = np.argsort(keys, kind="stable")
        keys, rows, fams = keys[order], rows[order], fams[order]
        uk, start, cnt = np.unique(keys, return_index=True, return_counts=True)
        keep = cnt <= C.KEY_DF_CAP
        self.ukeys = uk[keep]
        self.start = start[keep].astype(np.int64)
        self.cnt = cnt[keep].astype(np.int64)
        kf = fams[start[keep]]
        self.weight = (np.log1p(len(s1) / self.cnt) * FAM_W[kf]).astype(np.float32)
        self.kfam = kf
        self.rows = rows
        print(f"[index] {len(keys):,} postings, {len(uk):,} keys, {keep.sum():,} kept (df<={C.KEY_DF_CAP})")

    def query(self, qrows, qkeys, topk):
        pos = np.searchsorted(self.ukeys, qkeys)
        pos[pos >= len(self.ukeys)] = 0
        hit = self.ukeys[pos] == qkeys
        qrows, pos = qrows[hit], pos[hit]
        cnt = self.cnt[pos]
        tot = int(cnt.sum())
        rep = np.repeat(np.arange(len(pos)), cnt)
        offs = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        s1 = self.rows[self.start[pos][rep] + offs]
        q = qrows[rep]
        w = self.weight[pos][rep]
        fam = self.kfam[pos][rep]
        del rep, offs
        code = (q.astype(np.int64) << 22) | s1.astype(np.int64)
        uc, inv = np.unique(code, return_inverse=True)
        score = np.bincount(inv, weights=w).astype(np.float32)
        uq = (uc >> 22).astype(np.int32)
        us1 = (uc & ((1 << 22) - 1)).astype(np.int32)
        # top-k per query by score
        order = np.lexsort((-score, uq))
        uq_o = uq[order]
        first = np.r_[0, np.flatnonzero(np.diff(uq_o)) + 1]
        rank = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
        sel = order[rank < topk]
        rank_sel = rank[rank < topk]
        # per-family scores for the kept pairs only
        keep_pair = np.full(len(uc), -1, dtype=np.int64)
        keep_pair[sel] = np.arange(len(sel))
        kp = keep_pair[inv]
        m = kp >= 0
        fam_score = np.bincount(kp[m] * NF + fam[m], weights=w[m],
                                minlength=len(sel) * NF).astype(np.float32).reshape(len(sel), NF)
        return uq[sel], us1[sel], score[sel], rank_sel.astype(np.int16), fam_score


def iter_candidates(index, pool, dfq, topk=C.TOPK_RETRIEVE, qidx=None):
    """Yield candidate DataFrames chunk by chunk (bounded memory).
    qidx: optional subset of query rows to retrieve for."""
    sub = dfq if qidx is None else dfq.iloc[qidx]
    base = np.arange(len(dfq), dtype=np.int32) if qidx is None else np.asarray(qidx, dtype=np.int32)
    buf = []
    n = C.QUERY_CHUNK // C.N_JOBS + 1

    def flush():
        rows = np.concatenate([b[0] for b in buf]); keys = np.concatenate([b[2] for b in buf])
        buf.clear()
        q, s1, sc, rk, fs = index.query(rows, keys, topk)
        cand = pd.DataFrame({"q": base[q], "s1": s1, "ret_score": sc, "ret_rank": rk})
        for j, f in enumerate(FAMILIES):
            cand["k_" + f] = fs[:, j]
        return cand

    for part in emit_keys(sub[["country", "name_n", "addr_n", "nums"]], pool, index_side=False, n=n):
        buf.append(part)
        if len(buf) == C.N_JOBS:
            yield flush()
    if buf:
        yield flush()


def recall_report(cand, y, mask=None, ks=(1, 3, 5, 10, 20, 30)):
    """Recall of the true S1 among retrieved candidates, over matched queries."""
    m = y >= 0 if mask is None else (y >= 0) & mask
    hit = cand[cand.s1.values == y[cand.q.values]]
    best = np.full(len(y), 10_000, dtype=np.int32)
    best[hit.q.values] = hit.ret_rank.values
    n = m.sum()
    return {f"R@{k}": round(float((best[m] < k).sum() / n), 5) for k in ks}
