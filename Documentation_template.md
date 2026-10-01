# Amazon ML Challenge 2026 — Business Entity Resolution

## 1. Problem understanding
Match every S2/S3 business record to the S1 entity it describes, if any. Inputs are name, address and country. The output is, for each S1 entity, the list of matching S2/S3 ids.

The ground truth showed these structural facts:
- **Each S2/S3 record matches at most one S1 entity.** We therefore treat the task as assignment (choose the best S1 or none) rather than open pair classification. This removes the N² pair explosion and any risk of transitive chaining.
- **27% of train S2/S3 records match nothing.** The test S1 file is smaller relative to S2/S3 (2.82 S2 per S1 vs 2.28 in train), so about 40% of test records are "orphans" whose entity is absent.
- **Country always agrees between matches.** Parsed city agrees only 53% of the time.
- **54% of S1 names are shared by several S1 entities**, so name × address is the discriminative unit.
- **15% of S2 and 12% of S3 names are in native Indic scripts.**

## 2. Solution architecture
1. **Normalization.** Unidecode, OCR/leet repair (`c1inic`→`clinic`, `lnc`→`inc`), domain stripping (`x.com`), street-type and state canonicalization, and number extraction.
2. **Learned vocabularies (train fold only).**
   - Noise-token stoplist: legal forms, honorifics, and injected words such as `formerly` and `partners`.
   - Transliteration dictionary (Indic → English tokens, e.g. `praaivett`→`private`), learned from co-occurrence in matched pairs. It raised native-script retrieval recall from 82% to 99%.
3. **Blocking / candidate generation.**
   - Inverted index over 14 hashed key families: compact name, prefix, tokens, token pairs, phonetic skeleton, number×street, name×address combinations, and others.
   - IDF-weighted scoring, with keys dropped above a document-frequency cap (400).
   - Chunked processing keeps peak RAM bounded (about 12 GB).
   - Top-30 candidates per record: recall 98.9%.
4. **Pruner.** LightGBM on 30 cheap features (retrieval family scores and rapidfuzz ratios). Keeps the top 8 per record: recall 98.7%.
5. **Ranker.** LightGBM binary classifier on 90 features:
   - Name similarities: ratio, partial, token-sort, token-set, Jaro-Winkler, Levenshtein.
   - Compact-name and skeleton similarities.
   - Address similarities.
   - Number overlap and house-number difference.
   - State match and IDF-weighted token overlap.
   - Name-ambiguity count.
   - Query-relative gap and rank features.
6. **Assignment.** For each S2/S3 record, take the argmax S1 and accept it if p ≥ τ. Output is grouped by S1.

## 3. Validation
- 15% of S1 clusters are held out. Their matches are never used for any learned component.
- Validation per-record accuracy is 0.976 and pairwise F1 is 0.982.
- About 58% of the remaining errors are records with no address whose name is shared by several S1 entities. These cannot be resolved from content.

## 4. Threshold selection for F0.5 and distribution shift
The leaderboard metric is F0.5, which weights precision twice as much as recall. So τ is set conservatively.
The leaderboard showed that test contains many more orphans than train. Orphans whose names are generic match lookalike S1 entities with medium scores.

| τ | Leaderboard |
|---|---|
| 0.18 | 0.960 |
| 0.5 | 0.973 |
| 0.9 | **0.977** (final) |

A follow-up (v2) retrained on train with 19% of S1 removed, which reproduces the test orphan rate (`gt.orphan_view`), and re-scored the uncertain test band.

## 5. Files
- `output/matching_results.tsv`: final predictions, one row per test S1 (tau = 0.9, leaderboard 0.977).
- `output/candidate_pairs.tsv`: the full blocking candidate set, 79,037,916 pairs. It holds the top-8 S1 candidates per S2/S3 record after blocking and pruning; every pair was scored by the ranker. Columns: `source1_entity_id`, `candidate_entity_id`, `ranker_score`, `rank`.
- `code/business_entity_resolution/`: full source, trained models, README and requirements.

## 6. Tools
Python 3.14, pandas, pyarrow, numpy, rapidfuzz, unidecode, LightGBM (MIT). CPU only (16 cores, 24 GB RAM).

The final model is a LightGBM gradient-boosted tree ensemble (MIT license), far below the 8B-parameter limit. No pretrained neural models are used in the submitted pipeline. `er/neural.py` is an optional, unused module.
