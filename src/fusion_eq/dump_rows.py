"""Write the MAST input stream to per-shot pickles (NNNN.pkl, stream order) for mast_beta,
mast_vertical and `mast_batch --rows`, so post-hoc steps do not re-download the split.

    FUSION_SPLIT=private_test PYTHONPATH=src uv run python -m fusion_eq.dump_rows ROWS_DIR
"""
import argparse
import os
import pickle
from pathlib import Path

from .paths import DATA_REPO, DATA_REVISION, SPLIT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    from datasets import load_dataset
    a.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for n, row in enumerate(load_dataset(DATA_REPO, f"mast_{SPLIT}", split=SPLIT, streaming=True,
                                         revision=DATA_REVISION), 1):
        f = a.out / f"{n - 1:04d}.pkl"
        if not f.exists():  # resumable; the tmp rename keeps a killed run from leaving a torn row
            tmp = f.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(row, fh)
            os.replace(tmp, f)
    print("rows", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
