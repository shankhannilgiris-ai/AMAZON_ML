# AMAZON_ML
Entity resolution matching 10M business records to 1.7M entities on one CPU. Normalization, learned Indic transliteration, multi-key blocking (98.9% recall) and a LightGBM ranker. Each record goes to its best match, with a threshold tuned for F0.5. Leaderboard score: 0.977 (baseline 0.967).
