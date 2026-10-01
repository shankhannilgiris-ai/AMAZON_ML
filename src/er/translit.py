"""Learned transliteration dictionary for native-script (Indic) records.

~15% of S2 / ~12% of S3 names are written in Devanagari/Telugu/Kannada/...
unidecode gives 'praaivett limittedd'; the S1 twin says 'private limited'.
The generator draws from a finite business vocabulary, so we learn
translit-token -> english-token from co-occurrence in train-fold pairs
(Dice coefficient, tie-broken by string similarity) and rewrite native
records into English before blocking/features.
"""
import collections
import json
import numpy as np
from rapidfuzz.distance import JaroWinkler

from . import config as C

PATH = C.MODELS / "translit.json"


def _count(pairs):
    cq, ct, cqt = collections.Counter(), collections.Counter(), collections.Counter()
    for a, b in pairs:
        A = set(a.split()); B = set(b.split())
        cq.update(A); ct.update(B)
        for x in A:
            for y in B:
                cqt[(x, y)] += 1
    return cq, ct, cqt


def _build(cq, ct, cqt, min_count=2, min_sim=0.45, npairs=1):
    """For each transliterated token x choose english y maximising
    P(y|x) * (0.35 + JW(x, y)).  P(y|x) alone cannot separate words that always
    co-occur ('private'/'limited'); the string similarity can.  Very frequent,
    near-deterministic pairs are accepted even when dissimilar ('praa'->'pvt')."""
    by_q = collections.defaultdict(list)
    for (x, y), c in cqt.items():
        if cq[x] >= min_count:
            by_q[x].append((c / cq[x], y))
    out = {}
    for x, lst in by_q.items():
        lst.sort(reverse=True)
        pmax = lst[0][0]
        if pmax < 0.3:
            continue
        top = [(p, y) for p, y in lst[:8] if p >= 0.6 * pmax]
        # lift term stops ubiquitous targets ('limited') from absorbing rare
        # words that merely co-occur with them ('lkssmii' -> 'laxmi', not 'limited')
        score = lambda py: (py[0] * (0.35 + JaroWinkler.similarity(x, py[1]))
                            * np.sqrt(np.log1p(npairs / ct[py[1]])))
        best_p, best_y = max(top, key=score)
        sim = JaroWinkler.similarity(x, best_y)
        if best_y != x and (sim >= min_sim or (best_p >= 0.9 and cq[x] >= 50 and len(top) == 1)):
            out[x] = best_y
    return out


def learn(s1, queries, labels, train_mask_fn, max_pairs=2_000_000):
    name_pairs, addr_pairs = [], []
    for df, y, k in zip(queries, labels, (2, 3)):
        idx = np.where((y >= 0) & train_mask_fn(k) & (df.native.values == 1))[0][:max_pairs]
        name_pairs += list(zip(df.name_n.values[idx], s1.name_n.values[y[idx]]))
        # addresses: only when the query address itself is non-ascii
        ai = idx[~np.array([a.isascii() for a in df.addr_raw.values[idx]], dtype=bool)]
        addr_pairs += list(zip(df.addr_n.values[ai].astype(str), s1.addr_n.values[y[ai]]))
    name_map = _build(*_count(name_pairs), npairs=len(name_pairs))
    clean = lambda s: s.replace(",", " ")
    addr_map = _build(*_count([(clean(a), clean(b)) for a, b in addr_pairs]), min_count=5,
                      min_sim=0.75, npairs=len(addr_pairs))
    # never rewrite digits
    addr_map = {k: v for k, v in addr_map.items() if not k.isdigit() and not v.isdigit()}
    # English target vocabulary for fuzzy fallback on unseen transliterations
    vocab = collections.Counter(t for n in s1.name_n.values[s1.country.values == 1] for t in n.split())
    eng = sorted(t for t, c in vocab.items() if c >= 3 and len(t) >= 3)
    PATH.write_text(json.dumps({"name": name_map, "addr": addr_map, "eng": eng}))
    return name_map, addr_map


def extend_fuzzy(tokens, name_map, eng, cutoff=0.80):
    """Map unseen transliterated tokens to the closest English vocab word
    (in phonetic-skeleton space), vectorised via rapidfuzz.process.cdist."""
    from rapidfuzz import process
    from .text import skeleton
    todo = sorted({t for t in tokens if t not in name_map and len(t) >= 3})
    if not todo:
        return {}
    ek = [skeleton(t) for t in eng]
    qk = [skeleton(t) for t in todo]
    out = {}
    B = 20_000
    for i in range(0, len(todo), B):
        sim = process.cdist(qk[i:i + B], ek, scorer=JaroWinkler.normalized_similarity,
                            workers=-1, dtype=np.float32)
        j = sim.argmax(1); best = sim[np.arange(len(j)), j]
        for t, jj, b in zip(todo[i:i + B], j, best):
            if b >= cutoff:
                out[t] = eng[jj]
    return out


def load():
    d = json.loads(PATH.read_text())
    return d["name"], d["addr"], d["eng"]


def apply_name(s, m):
    return " ".join(m.get(t, t) for t in s.split())


def apply_addr(s, m):
    return ", ".join(" ".join(m.get(t, t) for t in comp.split()) for comp in s.split(", "))
