"""End-to-end orchestration. Every stage caches its output, so the pipeline is
resumable; delete a stage's output to recompute it.

train:  prepare -> learn -> retrieve(+prune) -> features -> train -> evaluate
test:   prepare -> retrieve(+prune) -> predict -> assign -> matching_results.tsv
"""
import gc
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C, prepare, gt, vocab, translit, data, retrieve as R, features as F, rank

TRAIN_QUERY_FRAC = 0.25       # fraction of train-fold queries used to fit the ranker
VAL_QUERY_FRAC = 1.0         # fraction of val queries scored (time budget)
PRUNER_SAMPLE = 150_000       # train-fold queries per source used to fit the pruner


def _pool_init(stop, stop_skel, idf):
    R._init(stop, stop_skel)
    F._init(stop, idf)


def _pool(s1):
    stop, stop_skel = vocab.load()
    return Pool(C.N_JOBS, initializer=_pool_init, initargs=(stop, stop_skel, F.token_idf(s1)))


def _log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------- learn
def stage_learn():
    if vocab.PATH.exists() and translit.PATH.exists():
        return
    s = [pd.read_parquet(C.cache_path("train", k),
                         columns=["eid", "country", "name_n", "addr_n", "addr_raw", "native"])
         for k in C.SOURCES]
    y2, y3, val = gt.load_labels(*s)
    masks = {2: ~gt.query_is_val(y2, val, 2), 3: ~gt.query_is_val(y3, val, 3)}
    vocab.learn(s[0], [s[1], s[2]], [y2, y3], lambda k: masks[k])
    translit.learn(s[0], [s[1], s[2]], [y2, y3], lambda k: masks[k])
    _log("learned noise vocabulary + transliteration dictionary")


def _train_view(columns=None):
    """Train data under the test-like orphan rate (see gt.orphan_view)."""
    S = data.load("train", columns=columns)
    y2, y3, val = gt.load_labels(*S)
    s1, y2, y3, val = gt.orphan_view(S[0], y2, y3, val)
    labels = {2: y2, 3: y3}
    isval = {2: gt.query_is_val(y2, val, 2), 3: gt.query_is_val(y3, val, 3)}
    return s1, S[1:], labels, isval, val


# ------------------------------------------------------------------ retrieve
def _part_dir(split, k, what):
    d = C.CACHE / f"{split}_{what}_s{k}"
    d.mkdir(exist_ok=True)
    return d


def _train_query_subset(y, isval, rng):
    tr = np.where(~isval)[0]
    tr = rng.choice(tr, int(len(tr) * TRAIN_QUERY_FRAC), replace=False)
    va = np.where(isval)[0]
    va = rng.choice(va, int(len(va) * VAL_QUERY_FRAC), replace=False)
    return np.sort(np.concatenate([tr, va]))


def stage_retrieve(split):
    done = C.CACHE / f"{split}_retrieve.done"
    if done.exists():
        return
    cols = ["eid", "country", "name_n", "addr_n", "nums", "addr_len", "native"]
    labels = None
    if split == "train":
        s1, qs, labels, isval, _ = _train_view(cols)
    else:
        S = data.load(split, columns=cols)
        s1, qs = S[0], S[1:]
    with _pool(s1) as pool:
        index = R.S1Index(s1, pool)
        _log("S1 index built")
        if split == "train" and not rank.PRUNER_PATH.exists():
            _fit_pruner(index, pool, s1, qs, labels, isval)
        pruner = lgb.Booster(model_file=str(rank.PRUNER_PATH))
        rng = np.random.default_rng(C.SEED)
        for k, Q in zip((2, 3), qs):
            if split == "train":
                qidx = _train_query_subset(labels[k], isval[k], rng)
            else:
                sub = C.CACHE / f"{split}_subset_s{k}.npy"      # optional: re-score only these queries
                qidx = np.load(sub) if sub.exists() else None
            out = _part_dir(split, k, "cand")
            hits = n_true = 0
            for i, cand in enumerate(R.iter_candidates(index, pool, Q, qidx=qidx)):
                cand = rank.prune_frame(cand, Q, s1)
                cand["prune_score"] = pruner.predict(cand[rank.PRUNE_FEATS].values, num_threads=C.N_JOBS).astype(np.float32)
                cand = rank.top_per_query(cand, "prune_score", C.TOPK_RANK)
                cand.to_parquet(out / f"part{i:04d}.parquet", index=False)
                if labels is not None:
                    y = labels[k]
                    hits += int((cand.s1.values == y[cand.q.values]).sum())
                _log(f"retrieve {split} s{k} chunk {i}: {len(cand):,} pairs")
            if labels is not None:
                qq = qidx if qidx is not None else np.arange(len(Q))
                n_true = int((labels[k][qq] >= 0).sum())
                _log(f"s{k} recall@{C.TOPK_RANK} after pruning: {hits / n_true:.5f}")
    done.touch()


def _fit_pruner(index, pool, s1, qs, labels, isval):
    rng = np.random.default_rng(C.SEED + 7)
    frames = []
    for k, Q in zip((2, 3), qs):
        tr = np.where(~isval[k])[0]
        va = np.where(isval[k])[0]
        pick = np.sort(np.concatenate([rng.choice(tr, PRUNER_SAMPLE, replace=False),
                                       rng.choice(va, PRUNER_SAMPLE // 4, replace=False)]))
        for cand in R.iter_candidates(index, pool, Q, qidx=pick):
            cand = rank.prune_frame(cand, Q, s1)
            cand["label"] = (cand.s1.values == labels[k][cand.q.values]).astype(np.int8)
            cand["is_val"] = isval[k][cand.q.values]
            cand["q"] = cand.q.values.astype(np.int64) + (k - 2) * 10_000_000   # keep S2/S3 query ids distinct
            frames.append(cand)
    df = pd.concat(frames, ignore_index=True)
    tr, va = df[~df.is_val], df[df.is_val]
    _log(f"pruner training rows {len(tr):,} / val {len(va):,}")
    m = rank.train_lgb(tr[rank.PRUNE_FEATS].values, tr.label.values, va[rank.PRUNE_FEATS].values,
                       va.label.values, rank.PRUNER_PATH, rounds=800, names=rank.PRUNE_FEATS,
                       params=dict(num_leaves=63, learning_rate=0.1))
    va = va.assign(p=m.predict(va[rank.PRUNE_FEATS].values))
    kept = rank.top_per_query(va, "p", C.TOPK_RANK)
    retrieved = va.groupby("q").label.max().sum()
    _log(f"pruner val: positives retrieved@30={retrieved:,}  kept@{C.TOPK_RANK}={kept.label.sum():,}"
         f"  ({kept.label.sum() / max(retrieved, 1):.5f})")


# ------------------------------------------------------------------ features
def stage_features(split):
    """train only: heavy features for the ranker's training/validation set."""
    done = C.CACHE / f"{split}_features.done"
    if done.exists():
        return
    s1, qs, labels, isval, _ = _train_view()
    with _pool(s1) as pool:
        for k, Q in zip((2, 3), qs):
            out = _part_dir(split, k, "feat")
            for p in sorted(_part_dir(split, k, "cand").glob("part*.parquet")):
                part = rank.heavy_frame(pd.read_parquet(p), Q, s1, pool)
                part["label"] = (part.s1.values == labels[k][part.q.values]).astype(np.int8)
                part["is_val"] = isval[k][part.q.values]
                part["src"] = np.int8(k)
                fc = rank.ranker_features(part)
                part[fc] = part[fc].astype(np.float16)
                part.to_parquet(out / p.name, index=False)
                _log(f"features {split} s{k} {p.name}: {len(part):,} rows")
    done.touch()


# --------------------------------------------------------------------- train
def stage_train():
    if rank.RANKER_PATH.exists():
        return
    # Preallocated float32 matrices: never hold the float16 frame and its
    # float32 copy of all ~30M rows at once. Early stopping uses a query
    # subsample of val; the full val set is scored later in stage_predict.
    paths = [p for k in (2, 3) for p in sorted(_part_dir("train", k, "feat").glob("*.parquet"))]
    masks = []
    for p in paths:
        m_ = pd.read_parquet(p, columns=["q", "is_val"])
        v = m_.is_val.values
        keepv = v & ((m_.q.values.astype(np.int64) * 2654435761 % 7) == 0)   # ~1/7 of val queries, all their candidates
        masks.append((~v, keepv))
    ntr = sum(int(a.sum()) for a, _ in masks); nva = sum(int(b.sum()) for _, b in masks)
    fc = rank.ranker_features(pd.read_parquet(paths[0]).head(1))
    _log(f"ranker: {ntr:,} train rows, {nva:,} early-stop rows, {len(fc)} features")
    Xtr = np.empty((ntr, len(fc)), np.float32); ytr = np.empty(ntr, np.int8)
    Xva = np.empty((nva, len(fc)), np.float32); yva = np.empty(nva, np.int8)
    i = j = 0
    for p, (a, b) in zip(paths, masks):
        part = pd.read_parquet(p, columns=fc + ["label"])
        x = part[fc].values
        na, nb = int(a.sum()), int(b.sum())
        Xtr[i:i + na] = x[a]; ytr[i:i + na] = part.label.values[a]; i += na
        Xva[j:j + nb] = x[b]; yva[j:j + nb] = part.label.values[b]; j += nb
        del part, x
    gc.collect()
    m = rank.train_lgb(Xtr, ytr, Xva, yva, rank.RANKER_PATH, names=fc)
    imp = pd.Series(m.feature_importance("gain"), index=fc).sort_values(ascending=False)
    (C.MODELS / "ranker_importance.csv").write_text(imp.to_csv())
    _log("ranker trained; top features:\n" + imp.head(25).to_string())


# ------------------------------------------------------------------- predict
def stage_predict(split):
    """Score pruned candidates -> (src, q, s1, p). For test, features are
    computed on the fly (never stored) to stay within disk budget."""
    dst = C.CACHE / f"{split}_scores.parquet"
    if dst.exists():
        return pd.read_parquet(dst)
    m = lgb.Booster(model_file=str(rank.RANKER_PATH))
    fc = m.feature_name()
    outs = []
    if split == "train":
        for k in (2, 3):
            for p in sorted(_part_dir(split, k, "feat").glob("*.parquet")):
                part = pd.read_parquet(p)
                part = part[part.is_val.values]
                p_ = m.predict(part[fc].values.astype(np.float32), num_threads=C.N_JOBS)
                outs.append(pd.DataFrame({"src": np.int8(k), "q": part.q.values, "s1": part.s1.values, "p": p_}))
    else:
        S = data.load(split)
        s1, qs = S[0], S[1:]
        with _pool(s1) as pool:
            for k, Q in zip((2, 3), qs):
                for p in sorted(_part_dir(split, k, "cand").glob("part*.parquet")):
                    part = rank.heavy_frame(pd.read_parquet(p), Q, s1, pool)
                    p_ = m.predict(part[fc].values.astype(np.float32), num_threads=C.N_JOBS)
                    outs.append(pd.DataFrame({"src": np.int8(k), "q": part.q.values, "s1": part.s1.values, "p": p_}))
                    _log(f"predict {split} s{k} {p.name}")
    sc = pd.concat(outs, ignore_index=True)
    sc.to_parquet(dst, index=False)
    return sc


# -------------------------------------------------------------------- assign
def assign(scores, tau):
    """Each S2/S3 record joins at most one S1 entity: argmax over its
    candidates, accepted when p >= tau."""
    best = scores.sort_values(["src", "q", "p"], ascending=[True, True, False]).drop_duplicates(["src", "q"])
    return best[best.p.values >= tau]


def _scored_mask(k, n):
    """Val queries that went through retrieval (the time-budget subsample)."""
    rng = np.random.default_rng(C.SEED)
    s1, qs, ys, isval, _ = _TV
    m = np.zeros(n, bool)
    for kk in (2, 3):
        idx = _train_query_subset(ys[kk], isval[kk], rng)
        if kk == k:
            m[idx] = True
    return m


_TV = None


def _chunk_covered(k, n, scores):
    """Only queries whose retrieval chunk finished (run was cut short)."""
    q = scores.q.values[scores.src.values == k]
    m = np.zeros(n, bool)
    m[: int(q.max()) + 1] = True
    return m


def evaluate(split="train"):
    """Validation under the test-like orphan rate. tau is chosen by per-record
    accuracy: on the leaderboard (tau .18 -> .960, .5 -> .973, .9 -> .977)
    that is the measure that moves the way the LB does."""
    global _TV
    scores = stage_predict(split)
    _TV = _train_view(["eid", "country"])
    s1, _, ys, isval, val = _TV
    best = scores.sort_values(["src", "q", "p"], ascending=[True, True, False]).drop_duplicates(["src", "q"])
    val_s1 = np.where(val)[0]
    res = {}
    for tau in (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97):
        ok = tot = tp = fp = fn = 0
        T, P = {}, {}
        for k in (2, 3):
            b = best[best.src.values == k]
            pred = np.full(len(ys[k]), -1, dtype=np.int64)
            acc = b.p.values >= tau
            pred[b.q.values[acc]] = b.s1.values[acc]
            m = isval[k] & _scored_mask(k, len(isval[k]))
            if not (best.src.values == k).any():
                continue                       # source not scored in this (time-budget) run
            m &= _chunk_covered(k, len(m), scores)
            y = ys[k][m]; pr = pred[m]
            ok += int((y == pr).sum()); tot += int(m.sum())
            tp += int(((pr >= 0) & (pr == y)).sum())
            fp += int(((pr >= 0) & (pr != y)).sum())
            fn += int(((y >= 0) & (pr != y)).sum())
            qv = np.where(m)[0]
            for qq in qv[ys[k][qv] >= 0]:
                T.setdefault(ys[k][qq], set()).add((k, qq))
            for qq in qv[pred[qv] >= 0]:
                P.setdefault(pred[qq], set()).add((k, qq))
        jac = np.mean([1.0 if not (T.get(s) or P.get(s)) else
                       len(T.get(s, set()) & P.get(s, set())) / len(T.get(s, set()) | P.get(s, set()))
                       for s in val_s1])
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res[tau] = dict(query_acc=round(ok / tot, 5), precision=round(prec, 5), recall=round(rec, 5),
                        f1=round(2 * prec * rec / max(prec + rec, 1e-9), 5), s1_jaccard=round(float(jac), 5))
        _log(f"tau={tau}: {res[tau]}")
    tau = max((t for t in res), key=lambda t: res[t]["query_acc"])
    res["best_tau"] = tau
    _log(f"best tau by per-record accuracy: {tau}")
    (C.OUT / "val_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def write_submission(split="test", tau=None):
    if tau is None:
        vm = C.OUT / "val_metrics.json"
        tau = 0.9 if not vm.exists() else json.loads(vm.read_text())["best_tau"]   # 0.9 = submitted (F0.5)
    scores = stage_predict(split)
    S = data.load(split, columns=["eid"])
    a = assign(scores, float(tau))
    eids = {2: S[1].eid.values, 3: S[2].eid.values}
    m = np.empty(len(a), dtype=object)
    for k in (2, 3):
        sel = a.src.values == k
        m[sel] = eids[k][a.q.values[sel]]
    a = a.assign(m=m)
    grp = a.groupby("s1").m.apply(lambda x: ",".join(sorted(x)))
    out = pd.DataFrame({"source1_entity_id": S[0].eid.values})
    out["matched_entity_ids"] = grp.reindex(np.arange(len(out))).fillna("").values
    path = C.OUT / "matching_results.tsv"
    out.to_csv(path, sep="\t", index=False)
    _log(f"wrote {path}: {len(out):,} S1 rows, {len(a):,} links, tau={tau}")
    write_candidate_pairs(split, scores, S)
    return path


def write_candidate_pairs(split, scores, S):
    """Blocking candidate set: every (S1, S2/S3) pair that survived blocking +
    pruning and was scored by the ranker (top-8 per S2/S3 record)."""
    import pyarrow as pa, pyarrow.csv as pc
    sc = scores.sort_values(["src", "q", "p"], ascending=[True, True, False])
    rank_ = sc.groupby(["src", "q"], sort=False).cumcount().values + 1
    cand = np.empty(len(sc), dtype=object)
    for k in (2, 3):
        m = sc.src.values == k
        cand[m] = S[k - 1].eid.values[sc.q.values[m]]
    t = pa.table({"source1_entity_id": S[0].eid.values[sc.s1.values], "candidate_entity_id": cand,
                  "ranker_score": np.round(sc.p.values, 6), "rank": rank_})
    path = C.OUT / "candidate_pairs.tsv"
    pc.write_csv(t, path, pc.WriteOptions(delimiter="	", quoting_style="none"))
    _log(f"wrote {path}: {len(sc):,} candidate pairs")


def run(split, stages):
    for st in stages:
        _log(f"=== {split}: {st}")
        if st == "prepare":
            prepare.run(split)
        elif st == "learn":
            stage_learn()
        elif st == "retrieve":
            stage_retrieve(split)
        elif st == "features":
            stage_features(split)
        elif st == "train":
            stage_train()
        elif st == "evaluate":
            evaluate(split)
        elif st == "predict":
            stage_predict(split)
        elif st == "submit":
            write_submission(split)
