"""Stage 1 - load TSVs, normalise text in parallel, cache as parquet.

Row order inside each cache file defines the integer id used everywhere
downstream (s1 idx / query idx), so nothing ever joins on string ids.
"""
from multiprocessing import Pool
import numpy as np
import pandas as pd

from . import config as C
from .text import norm_name, norm_addr

COLS = ["entity_id", "business_name", "business_address", "country"]


def _names(chunk):
    return [norm_name(x) for x in chunk]


def _addrs(chunk):
    return [norm_addr(x) for x in chunk]


def _par(fn, values, pool, n=200_000):
    chunks = [values[i:i + n] for i in range(0, len(values), n)]
    out = []
    for part in pool.imap(fn, chunks):
        out.extend(part)
    return out


def prepare(split: str, src: int, pool) -> pd.DataFrame:
    dst = C.cache_path(split, src)
    if dst.exists():
        return pd.read_parquet(dst)
    df = pd.read_csv(C.raw_path(split, src), sep="\t", dtype=str, usecols=COLS,
                     engine="pyarrow")
    df = df.fillna("")
    names = _par(_names, df.business_name.tolist(), pool)
    addrs = _par(_addrs, df.business_address.tolist(), pool)
    out = pd.DataFrame({
        "eid": df.entity_id.values,
        "country": df.country.map(C.COUNTRY_CODE).fillna(2).astype(np.int8).values,
        "name_raw": df.business_name.values,
        "addr_raw": df.business_address.values,
        "name_n": [x[0] for x in names],
        "dom": np.array([x[1] for x in names], dtype=np.int8),
        "native": np.array([x[2] for x in names], dtype=np.int8),
        "addr_n": [x[0] for x in addrs],
        "nums": [x[1] for x in addrs],
        "state": [x[2] for x in addrs],
    })
    out.to_parquet(dst, index=False)
    return out


def run(split: str):
    with Pool(C.N_JOBS) as pool:
        for s in C.SOURCES:
            d = prepare(split, s, pool)
            print(f"[prepare] {split} s{s}: {len(d):,} rows")
