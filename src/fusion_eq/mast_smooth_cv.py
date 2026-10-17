"""Label-free choice of the MAST time-smoothing windows by leave-one-out cross-validation.

A running mean of w frames trades noise for lag. Predicting each frame from its w - 1 neighbours
and scoring against the frame itself estimates that trade with no target: independent frame
noise adds the same variance to every window's error, so the window with the lowest
leave-one-out error is also the one closest to the underlying signal.

The noise enters through Thomson scattering, so the CV runs on the Thomson time base, where
frames are separate laser pulses with independent noise. On EFIT frames it would not: those are
interpolated between Thomson pulses, neighbours share noise, and LOO would favour short windows.
- BETA_SMOOTH: the volume-averaged Thomson pressure, which carries all the frame-to-frame scatter
  of betaN (Ip, B0 and the minor radius are smooth by comparison).
- SMOOTH (psi, q95): the Thomson outer edge, which positions every free-boundary solve.
The window is chosen on the median over shots of each shot's normalised LOO error (the pooled
error is also printed), and its span in ms is converted to the nearest odd count of EFIT frames. Uses MAST inputs only (Rule 3: label-free
test-time adaptation); no equilibrium label is read.

    PYTHONPATH=src:vendor/starter uv run python -m fusion_eq.mast_smooth_cv [ROWS_DIR]
"""
import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

WINDOWS = np.arange(3, 32, 2)  # Thomson frames, odd so the window is centred


def loo_errors(x, windows=WINDOWS) -> tuple[np.ndarray, np.ndarray]:
    """(sum of squared LOO errors, count) per window for one series, over the frames where every
    window fits, so all windows are scored on the same frames. NaN frames are left out."""
    h = int(windows.max()) // 2
    sse, n = np.zeros(len(windows)), np.zeros(len(windows))
    if len(x) < 2 * h + 1:
        return sse, n
    c = np.arange(h, len(x) - h)
    for i, w in enumerate(windows):
        k = int(w) // 2
        nb = np.stack([x[c + d] for d in range(-k, k + 1) if d])
        pred = np.nanmean(nb, axis=0) if np.isfinite(nb).any() else np.full(len(c), np.nan)
        e = (x[c] - pred) ** 2
        ok = np.isfinite(e)
        sse[i], n[i] = e[ok].sum(), ok.sum()
    return sse, n


def thomson_series(row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Thomson times ms, outer edge m, mean pressure / its shot median) per pulse, NaN where a
    pulse has no profile: interpolated pulses would share noise with their neighbours."""
    from .mast import mean_pressure, thomson_edges
    tt = np.asarray(row["thomson_core_times"], float)
    rin, rout = thomson_edges(row, tt, raw=True)
    with np.errstate(all="ignore"):
        p = mean_pressure(row, tt, rin, rout, raw=True)
        p = p / np.nanmedian(p) if np.isfinite(p).any() else p
    return tt, rout, p


def _one(path):
    row = pickle.load(open(path, "rb"))
    tt, rout, p = thomson_series(row)
    dt = np.median(np.diff(tt[tt > 1])) if (tt > 1).sum() > 2 else np.nan  # pulse 0 sits at ~0 ms
    efit_dt = np.median(np.diff(np.asarray(row["efit_times"], float)))
    return loo_errors(rout), loo_errors(p), dt, efit_dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="?", default="/Volumes/PortableSSD/scratch/fusion-eq-runs/mast/rows")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    files = sorted(Path(a.rows).glob("*.pkl"))
    with Pool(a.workers) as pool:
        res = pool.map(_one, files, chunksize=8)
    dt = np.nanmedian([r[2] for r in res])
    efit_dt = np.nanmedian([r[3] for r in res])
    print(f"{len(files)} shots; Thomson pulse {dt:.2f} ms, EFIT frame {efit_dt:.2f} ms")
    for name, j in (("edge (SMOOTH)", 0), ("pressure (BETA_SMOOTH)", 1)):
        sse = sum(r[j][0] for r in res)
        n = sum(r[j][1] for r in res)
        cv = sse / np.maximum(n, 1)
        print(f"{name}: pooled LOO MSE " + " ".join(f"{w}:{v:.4g}" for w, v in zip(WINDOWS, cv)))
        # Per shot, normalised by the shot's mean over windows, so a few high-scatter shots do
        # not set the window for all; the chosen window is the median over shots.
        per = np.array([r[j][0] / np.maximum(r[j][1], 1) for r in res if r[j][1].min() > 20])
        norm = np.median(per / per.mean(1, keepdims=True), 0)
        print(f"  per-shot median ({len(per)} shots) " + " ".join(f"{w}:{v:.4f}" for w, v in zip(WINDOWS, norm)))
        for lab, v in (("pooled", cv), ("per-shot", norm)):
            best = int(WINDOWS[np.argmin(v)])
            frames = max(1, int(2 * round((best * dt / efit_dt - 1) / 2) + 1))
            print(f"  {lab} best {best} pulses = {best * dt:.0f} ms -> {frames} EFIT frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
