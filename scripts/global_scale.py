"""Derives the global twin scale c (0.39, `twin blend --c`): the least-squares slope of the DIII-D
EFIT residual (EFIT - iter4) on the scale-1 twin correction (twin - iter4), pooled over every
frame and grid point of one bench half. Same data and fit as `twin modes`, one slope instead of
one per PCA mode. Labels are DIII-D train EFIT only.
    PYTHONPATH=src python scripts/global_scale.py HALF_FILES BENCH_DIR
"""
import sys
from pathlib import Path

import numpy as np

from fusion_eq.d3d_bench import _init, frames_of, load_row


def global_scale(files, bench):
    rd = dd = 0.0
    for f in [l.strip() for l in open(files) if l.strip()]:
        st = Path(f).name[:-8]
        p0 = np.load(Path(bench) / "iter4" / f"{st}.npz")["psirz"].astype(np.float64)
        p1 = np.load(Path(bench) / "twin" / f"{st}.npz")["psirz"].astype(np.float64)
        row = load_row(f)
        gt = np.stack([np.asarray(row["efit_psirz"][k], float) for k in frames_of(len(row["efit_times"]))])
        # per-frame mean removed, as in `twin modes`: a flux offset does not change the equilibrium
        r, d = (gt - p0).reshape(len(gt), -1), (p1 - p0).reshape(len(gt), -1)
        r, d = r - r.mean(1, keepdims=True), d - d.mean(1, keepdims=True)
        rd += (r * d).sum()
        dd += (d * d).sum()
    return rd / dd


if __name__ == "__main__":
    _init()
    print(f"global scale {global_scale(sys.argv[1], sys.argv[2]):.3f}")
