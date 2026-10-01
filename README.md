# AMAZON_ML — Business Entity Resolution (Amazon ML Challenge 2026)

Entity resolution matching 10M business records to 1.7M entities on one CPU. Normalization, learned Indic transliteration, multi-key blocking (98.9% recall) and a LightGBM ranker. Each record goes to its best match, with a threshold tuned for F0.5. **Leaderboard score: 0.977** (baseline 0.967).

Full methodology: [Documentation_template.md](Documentation_template.md)

## Layout
```
src/
├── run.py              CLI: python run.py train | test [--stages ...]
├── er/                 pipeline package (see "Architecture" below)
└── models/             trained artifacts: noise stoplist, transliteration dictionary, pruner, ranker
```

## Setup
```bash
pip install -r requirements.txt
```
Put the `*_processed.tsv` files and `train_ground_truth.tsv` in `src/`, or edit `RAW` in `er/config.py`.

```bash
cd src
python run.py train      # prepare -> learn -> retrieve -> features -> train -> evaluate
python run.py test       # prepare -> retrieve -> predict -> submit  => output/matching_results.tsv
```
To reproduce the submitted file with the shipped models in `src/models/`, run `python run.py test` (no training needed). It uses tau = 0.9 and writes both `output/matching_results.tsv` and `output/candidate_pairs.tsv` (the blocking candidate set).

Hardware: Windows, 16 cores, 24 GB RAM, CPU only. Runtime: train ~1.5 h, test ~4.5 h.

## Key facts about the data (these shape the whole design)

Measured on `train_*`:

| Fact | Consequence |
|---|---|
| Every S2/S3 record belongs to **at most one** S1 entity (7.64M matched ids, none repeated) | Treat the task as *assignment*: for each S2/S3 record, pick the best S1 or none. There's no N² pair space and no transitive-closure chaining risk. |
| 27% of S2/S3 records are distractors (match nothing); 123k S1 have zero matches | A per-record reject threshold `tau` is required. |
| Country always agrees between matches | Hard-partition all blocking keys by country. |
| Parsed `city` agrees in only 53% of true pairs; `pincode` is mostly empty | City/pincode blocking (the old notebook) loses recall, so blocking uses raw normalized text. |
| 54% of true S1 names are shared by other S1 entities (24% by 20+) | Name alone doesn't identify an entity; **name × address** keys and features carry the signal. |
| 15% of S2 / 12% of S3 names are in native Indic script | A learned transliteration dictionary took native-name retrieval from 81% to 99% R@30. |
| Generator noise: legal-form shuffles, `Partners/Services/Center` injections, `Formerly X`, `x.com` domains, OCR/leet swaps (`c1inic`, `5ervices`, `lnc`), homoglyphs, typos, address component reordering, house number ±1, state name/abbr/native script | Handled in normalization, the learned stoplist and fuzzy features. |
| ~4.4% of queries have no address; ~2.2% have no address **and** a shared name | Content can't resolve these, so they cap achievable accuracy. |

## Architecture

```
TSV ──► [1] prepare: unidecode, de-leet, domain strip, address canonicalization ─► parquet cache
        [learn] noise stoplist + transliteration dictionary (TRAIN FOLD ONLY)
        [2] multi-pass inverted-index blocking (14 key families, IDF-weighted, df-capped)
             └─ top-30 S1 per S2/S3 record, per-family scores
        [3] pruner (LightGBM, 30 cheap features) ─► top-8 per record    (inside each chunk)
        [4] heavy features (27 rapidfuzz + 16 token/number + query-relative gap/rank)
        [5] LightGBM ranker (binary, grouped-by-S1 validation)
        [6] assignment: argmax per S2/S3 record, accept if p >= tau (tuned on val)
        [7] matching_results.tsv  (source1_entity_id, matched_entity_ids)
optional GPU (er/neural.py):
        [7'] fine-tuned bge-small bi-encoder: dense top-20 unioned into candidates
        [8'] fine-tuned MiniLM cross-encoder on contested queries, blended with the ranker
```

### Blocking key families (`er/retrieve.py`)

| family | example key | purpose |
|---|---|---|
| compact | `halcyonproductions` | exact core name; also catches `halcyonproductions.com` |
| prefix | `halcy` | truncation / suffix noise |
| tok, tokpair | `renaud`, `buckley\|mayes` | token shuffles, extra/dropped tokens |
| skel, skelpair | `mrktng` | phonetic skeleton (India) |
| numstreet | `1901\|eastbridge` (S1 side also 1900/1902) | pseudo-random names; house number ±1 |
| addrpair, addr_exact | `dayton\|hillcrest` | address-only matches |
| tok_num | `3400\|halcyon` | |
| name_addr, pre_addr, suf_addr, skel_addr | `realinvestment\|bangalore` | the discriminative unit for generic names |

Score = Σ IDF(key) × family weight. Keys shared by more than 400 S1 records are dropped, which prevents the pair explosion behind the old `MemoryError`. Peak RAM is bounded by `QUERY_CHUNK`, not dataset size.

## Repository structure

```
├── README.md
├── Documentation_template.md   methodology write-up
├── requirements.txt            pinned dependencies
└── src/
    ├── run.py                  CLI: python run.py train | test [--stages ...]
    ├── er/
    │   ├── config.py           paths, K values, chunk sizes, val fraction
    │   ├── text.py             normalization, de-leet, skeleton, address canonicalization
    │   ├── prepare.py          stage 1 (parallel), parquet cache
    │   ├── gt.py               ground truth → int label arrays, grouped train/val split
    │   ├── vocab.py            learned noise-token stoplist
    │   ├── translit.py         learned Indic → English token dictionary
    │   ├── data.py             loads cache + enrichment + derived columns
    │   ├── retrieve.py         stage 2 inverted-index blocking
    │   ├── features.py         cheap / heavy / query-group features
    │   ├── rank.py             pruner + ranker (LightGBM)
    │   ├── pipeline.py         stage orchestration, evaluation, submission
    │   └── neural.py           optional bi-encoder / cross-encoder (GPU, unused in submission)
    └── models/                 name_stop.json, translit.json, pruner.txt, ranker.txt
```

Generated at run time (git-ignored): `src/data/cache/`, `src/output/`, `src/logs/`. The challenge data is not included in this repository.

## Running

```bash
python run.py train     # prepare → learn → retrieve → features → train → evaluate
python run.py test      # prepare → retrieve → predict → submit  → output/matching_results.tsv
```

Every stage caches its output and is resumable. To recompute a stage, delete its artifact: `data/cache/<split>_retrieve.done`, `_features.done`, `models/ranker.txt`, and so on.

## Validation protocol

- 15% of **S1 clusters** are held out. A held-out S1 and all its S2/S3 matches are never seen by the stoplist, the transliteration dictionary, the pruner or the ranker.
- Distractors are split randomly with the same fraction.
- Reported metrics: per-query assignment accuracy, pairwise precision/recall/F1, and per-S1 exact-set accuracy (`output/val_metrics.json`).

## Validated results (held-out 15% of S1 clusters)

| stage | metric | S2 | S3 |
|---|---|---|---|
| blocking, no transliteration | R@30 (native-script queries) | 0.816 | 0.833 |
| blocking + learned transliteration | R@30 (native-script queries) | 0.991 | 0.991 |
| blocking (all queries) | R@30 | 0.989 | 0.988 |
| after pruner | R@8 | 0.988 | 0.986 |

Ranker and assignment (val queries, τ = 0.5):

| query-level accuracy | pairwise precision | pairwise recall | pairwise F1 | per-S1 exact-set accuracy |
|---|---|---|---|---|
| 0.9764 | 0.9924 | 0.9715 | **0.9818** | 0.9066 |

Where the remaining errors are:

| | queries with address (95.6%) | queries without address (4.4%) |
|---|---|---|
| correct | 98.97% | 58.9% |
| not retrieved | 0.35% | 19.1% |
| wrong S1 chosen | 0.11% | 15.9% |
| rejected (p < τ) | 0.32% | 5.7% |

58% of all errors are no-address records. They're typically generic names shared by 2–400 S1 entities, and no content signal separates them. Row order and entity-id numbers were checked for leakage and carry none. That sets a content-based ceiling of about 98.5% query accuracy. Reaching 0.99+ therefore depends on how the leaderboard metric is defined.

## Memory and runtime

| stage | peak RAM | time (16 cores) |
|---|---|---|
| prepare | ~6 GB | 1.5 min / split |
| S1 index (110M postings) | ~8 GB | 2.3 min |
| retrieve + prune | ~12 GB | ~60 s per 80k queries (train subset ~50 min, full test ~2 h) |
| heavy features | ~10 GB | see logs |
| ranker training | ~10 GB | see logs |

Bottlenecks and fixes already applied:
- **Pair explosion**: the df cap plus chunked `searchsorted` expansion replaces the Python dict-of-sets.
- **Pickling cost**: workers hash keys (`pd.util.hash_array`, which is deterministic across processes) and return only `uint64` arrays.
- **Disk**: only 24 GB was free, so test features are computed on the fly and never stored.
- **Next speedup if needed**: the per-chunk LightGBM pruner predict and pandas groupbys dominate retrieval time. Moving `_gap_rank` to numpy segment ops and using `num_threads` pinning would roughly halve it.
