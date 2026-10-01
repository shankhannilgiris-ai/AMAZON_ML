"""Amazon ML Challenge 2026 - entity resolution pipeline.

    python run.py train      # prepare, learn, retrieve, features, train, evaluate
    python run.py test       # prepare, retrieve, predict, submit
    python run.py train --stages retrieve features
"""
import argparse

from er import pipeline

DEFAULT = {
    "train": ["prepare", "learn", "retrieve", "features", "train", "evaluate"],
    "test": ["prepare", "retrieve", "predict", "submit"],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--stages", nargs="*")
    a = ap.parse_args()
    pipeline.run(a.split, a.stages or DEFAULT[a.split])
