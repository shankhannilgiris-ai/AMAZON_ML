"""Learn the name 'noise vocabulary' from ground-truth pairs.

A token is noise when the data generator adds/drops it often relative to how
often it appears (legal forms, honorifics, 'formerly', 'dba', generic
descriptors such as 'group'/'services').  Learned on the train fold only.
"""
import collections
import json
import numpy as np

from . import config as C
from .text import BASE_NAME_STOP, skeleton

PATH = C.MODELS / "name_stop.json"


def learn(s1, queries, labels, train_mask_fn, min_count=150, ratio=0.22, max_pairs=1_500_000):
    added, dropped, freq = collections.Counter(), collections.Counter(), collections.Counter()
    n1 = s1.name_n.values
    for t in n1:
        freq.update(set(t.split()))
    for df, y, k in zip(queries, labels, (2, 3)):
        # native-script records are handled in skeleton space; their
        # transliterated tokens are not 'noise' words
        idx = np.where((y >= 0) & train_mask_fn(k) & (df.native.values == 0))[0]
        idx = idx[: max_pairs // 2]
        qn = df.name_n.values
        for q in idx:
            a = set(qn[q].split()); b = set(n1[y[q]].split())
            added.update(a - b); dropped.update(b - a)
    stop = set(BASE_NAME_STOP)
    for t in set(added) | set(dropped):
        ch = added[t] + dropped[t]
        if ch < min_count:
            continue
        if len(t) == 1 or ch / (freq[t] + added[t] + 1) > ratio:
            stop.add(t)
    stop_skel = {skeleton(t) for t in stop if len(skeleton(t)) >= 2}
    PATH.write_text(json.dumps({"stop": sorted(stop), "stop_skel": sorted(stop_skel)}))
    return stop, stop_skel


def load():
    d = json.loads(PATH.read_text())
    return set(d["stop"]), set(d["stop_skel"])
