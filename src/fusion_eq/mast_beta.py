"""Recompute betaN on a solved MAST cache from the MAST inputs, without re-solving any equilibrium.

betaN comes from the Thomson edges and pressure, Ip and the TF current (mast.mast_scalars), never
from the solved flux, so a new BETA_SMOOTH reaches every cached solver variant at once.
Recomputed files carry `beta_win`; files already at the requested window are skipped.

    PYTHONPATH=src uv run python -m fusion_eq.mast_beta CACHE_DIR [--rows ROWS_DIR] [--win N]
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from .mast import BETA_SMOOTH, FAST_ION_FRAC, mast_scalars, smooth_frames, thomson_edges, total_beta_n


def beta_n(row, win: int = BETA_SMOOTH, frac: float = 0.3, ti_over_te: float = 1.0,
           fast_ion_frac: float = FAST_ION_FRAC) -> np.ndarray:
    """predict_mast's betaN at its defaults (kappa and delta enter q95 only)."""
    t = np.asarray(row["efit_times"], float)
    rin, rout = thomson_edges(row, t, frac)
    _, b = mast_scalars(row, t, rin, rout, 1.8, 0.4, ti_over_te)
    return smooth_frames(total_beta_n(b, fast_ion_frac), win)


def recompute(cache: Path, rows: Path, win: int = BETA_SMOOTH, fn=beta_n) -> dict:
    counts = {}
    for f in sorted(cache.glob("shot_[0-9][0-9][0-9][0-9].npz")):
        d = dict(np.load(f))
        if str(d["method"]) == "const":
            how = "const"
        elif int(d.get("beta_win", -1)) == win:
            how = "current"
        else:
            row = pickle.load(open(rows / f"{f.stem[5:]}.pkl", "rb"))
            d.update(betaN=fn(row, win).astype(np.float32), beta_win=np.array(win))
            tmp = f.with_name(f.stem + ".tmp.npz")
            np.savez(tmp, **d)
            os.replace(tmp, f)  # atomic, as in mast_batch
            how = "recomputed"
        counts[how] = counts.get(how, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--rows", default="/Volumes/PortableSSD/scratch/fusion-eq-runs/mast/rows")
    ap.add_argument("--win", type=int, default=BETA_SMOOTH)
    a = ap.parse_args()
    print(recompute(Path(a.cache), Path(a.rows), a.win))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
