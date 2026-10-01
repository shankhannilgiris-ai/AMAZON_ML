"""Phases 7-8 (optional, GPU): fine-tuned bi-encoder retrieval and
cross-encoder re-ranking.

Requires:  pip install torch --index-url https://download.pytorch.org/whl/cu128
           pip install sentence-transformers
Runs on a 6 GB laptop GPU (fp16, small models). CPU fallback works but is
~20x slower, so on CPU only use the cross-encoder on contested queries.

Design
- Bi-encoder (BAAI/bge-small-en-v1.5, 33M params) fine-tuned with in-batch
  negatives (MultipleNegativesRankingLoss) on (query record, true S1 record)
  pairs from the TRAIN FOLD. Text = "name | address" after transliteration.
  Exact kNN by fp16 matmul on GPU, per country (India S1 ~0.9M x 384 fp16 =
  0.7 GB). Output: top-20 dense candidates per query -> unioned with the
  sparse candidates before pruning (adds d_score / d_rank features).
- Cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) fine-tuned as a binary
  classifier on hard negatives = the pruned candidates. Applied only where
  the LightGBM ranker is uncertain (top-1 p < 0.97 or top1-top2 margin < 0.3),
  typically 5-10% of queries, then blended with the ranker by a logistic
  regression fitted on validation (ensemble.py).
"""
import numpy as np
import pandas as pd

from . import config as C

BI_MODEL = "BAAI/bge-small-en-v1.5"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BI_DIR = C.MODELS / "biencoder"
CE_DIR = C.MODELS / "crossencoder"


def record_text(df):
    return (df.name_n.fillna("") + " | " + df.addr_n.fillna("")).tolist()


# ------------------------------------------------------------- bi-encoder
def train_biencoder(s1, queries, labels, train_mask_fn, n_pairs=1_500_000, epochs=1, bs=256):
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    t1 = record_text(s1)
    ex = []
    rng = np.random.default_rng(C.SEED)
    for df, y, k in zip(queries, labels, (2, 3)):
        idx = np.where((y >= 0) & train_mask_fn(k))[0]
        idx = rng.choice(idx, min(n_pairs // 2, len(idx)), replace=False)
        tq = record_text(df.iloc[idx])
        ex += [InputExample(texts=[a, t1[s]]) for a, s in zip(tq, y[idx])]
    model = SentenceTransformer(BI_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = 64
    dl = DataLoader(ex, shuffle=True, batch_size=bs, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)   # in-batch negatives
    model.fit(train_objectives=[(dl, loss)], epochs=epochs, warmup_steps=500,
              use_amp=True, show_progress_bar=True)
    model.save(str(BI_DIR))
    return model


def encode(model, texts, bs=1024):
    return model.encode(texts, batch_size=bs, convert_to_tensor=True, normalize_embeddings=True,
                        show_progress_bar=True).half()


def dense_retrieve(split, s1, queries, topk=20, qbatch=8192):
    """Exact inner-product kNN per country on GPU. Writes {split}_dense_s{k}.parquet."""
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(BI_DIR), device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = 64
    for k, Q in zip((2, 3), queries):
        dst = C.CACHE / f"{split}_dense_s{k}.parquet"
        if dst.exists():
            continue
        outs = []
        for c in (0, 1):
            si = np.where(s1.country.values == c)[0]
            qi = np.where(Q.country.values == c)[0]
            E1 = encode(model, record_text(s1.iloc[si]))
            for b in range(0, len(qi), qbatch):
                qq = qi[b:b + qbatch]
                Eq = encode(model, record_text(Q.iloc[qq]))
                sim = Eq @ E1.T
                v, j = torch.topk(sim, topk, dim=1)
                outs.append(pd.DataFrame({
                    "q": np.repeat(qq, topk).astype(np.int32),
                    "s1": si[j.cpu().numpy().ravel()].astype(np.int32),
                    "d_score": v.float().cpu().numpy().ravel(),
                    "d_rank": np.tile(np.arange(topk, dtype=np.int16), len(qq))}))
            del E1
            torch.cuda.empty_cache()
        pd.concat(outs, ignore_index=True).to_parquet(dst, index=False)


# ---------------------------------------------------------- cross-encoder
def train_crossencoder(pairs_df, s1, queries, n=3_000_000, epochs=1, bs=128):
    """pairs_df: pruned candidates of TRAIN-FOLD queries with columns src,q,s1,label."""
    import torch
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader
    df = pairs_df.sample(min(n, len(pairs_df)), random_state=C.SEED)
    t1 = record_text(s1)
    tq = {2: record_text(queries[0]), 3: record_text(queries[1])}
    ex = [InputExample(texts=[tq[k][q], t1[s]], label=float(l))
          for k, q, s, l in zip(df.src.values, df.q.values, df.s1.values, df.label.values)]
    ce = CrossEncoder(CE_MODEL, num_labels=1, max_length=128,
                      device="cuda" if torch.cuda.is_available() else "cpu")
    ce.fit(train_dataloader=DataLoader(ex, shuffle=True, batch_size=bs), epochs=epochs,
           warmup_steps=1000, use_amp=True, show_progress_bar=True)
    ce.save(str(CE_DIR))
    return ce


def contested(scores, p_hi=0.97, margin=0.3):
    """Queries where the ranker is not confident -> send to the cross-encoder."""
    s = scores.sort_values(["src", "q", "p"], ascending=[True, True, False])
    g = s.groupby(["src", "q"], sort=False).p
    top1 = g.transform("max")
    second = g.transform(lambda x: x.iloc[1] if len(x) > 1 else 0.0)
    m = (top1 < p_hi) | ((top1 - second) < margin)
    return s[m.values]


def crossencode(pairs, s1, queries, bs=512):
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder
    ce = CrossEncoder(str(CE_DIR), max_length=128, device="cuda" if torch.cuda.is_available() else "cpu")
    t1 = record_text(s1)
    tq = {2: record_text(queries[0]), 3: record_text(queries[1])}
    txt = [[tq[k][q], t1[s]] for k, q, s in zip(pairs.src.values, pairs.q.values, pairs.s1.values)]
    return ce.predict(txt, batch_size=bs, show_progress_bar=True, convert_to_numpy=True)
