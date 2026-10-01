"""Load cached sources with learned enrichments applied.

- native records: name_n / addr_n rewritten through the learned
  transliteration dictionary (pre-transliteration text kept in name_u)
- derived helper columns for features (compact, skel, addr_first, ...)
- S1 only: `dup` = number of S1 records sharing (country, core name)
"""
import numpy as np
import pandas as pd

from . import config as C, translit, vocab
from .features import add_derived_columns


def _enrich(df, name_map, addr_map, stop):
    df = df.copy()
    df["name_u"] = df.name_n
    nat = df.native.values == 1
    df.loc[nat, "name_n"] = [translit.apply_name(x, name_map) for x in df.name_n.values[nat]]
    ind = df.country.values == 1
    df.loc[ind, "addr_n"] = [translit.apply_addr(x, addr_map) for x in df.addr_n.values[ind]]
    df = df.drop(columns=["name_raw", "addr_raw"])
    return add_derived_columns(df, stop)


def load(split, columns=None):
    out = []
    name_map, addr_map, _ = translit.load()
    stop, _ = vocab.load()
    for k in C.SOURCES:
        p = C.CACHE / f"{split}_s{k}_enr.parquet"
        if not p.exists():
            df = _enrich(pd.read_parquet(C.cache_path(split, k)), name_map, addr_map, stop)
            if k == 1:
                key = df.country.astype(str) + "|" + df.compact
                df["dup"] = key.map(key.value_counts()).astype(np.int32).values
            df.to_parquet(p, index=False)
        cols = columns
        if cols is not None and k != 1:
            cols = [c for c in cols if c != "dup"]
        out.append(pd.read_parquet(p, columns=cols))
    return out
