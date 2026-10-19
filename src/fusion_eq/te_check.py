"""Label-free psi check (python -m fusion_eq.te_check FILES VARIANT...): Te is a flux function, so Te must be monotone in psi over the Thomson
chord points. Per frame, 1 - |Spearman(Te, psi)|; validated against EFIT truth on DIII-D bench."""
import os
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr

from .d3d_bench import load_row

BENCH = Path("../benchA")


def thomson_points(row, t, dt=8.0):
    """(R, Z, Te) of chord points with Te>0 from the core+tangential pulse nearest t (ms)."""
    names = [str(x) for x in row["thomson_chord_name"]]
    R = np.asarray(row["thomson_chord_R"], float); Z = np.asarray(row["thomson_chord_Z"], float)
    out = []
    kinds = (("core", "thomson_core"), ("tangential", "thomson_edge"))
    if os.environ.get("TE_MIDPLANE"):  # the MAST-like geometry: radial midplane chord only
        kinds = kinds[1:]
    for kind, key in kinds:
        idx = [i for i, n in enumerate(names) if f"_{kind}_" in n]
        tt = np.asarray(row[key + "_times"], float); Te = np.asarray(row[key + "_Te"], float)
        good = np.where((Te > 0).sum(1) >= 3)[0]
        if not len(good):
            continue
        j = good[np.abs(tt[good] - t).argmin()]
        if abs(tt[j] - t) > dt:
            continue
        m = Te[j] > 0
        out.append(np.c_[R[idx][m], Z[idx][m], Te[j][m]])
    return np.vstack(out) if out else np.zeros((0, 3))


def frame_err(psi, Rg, Zg, pts):
    if len(pts) < 6:
        return np.nan
    f = RegularGridInterpolator((Zg, Rg), psi, bounds_error=False, fill_value=np.nan)
    p = f(np.c_[pts[:, 1], pts[:, 0]])
    ok = np.isfinite(p)
    if ok.sum() < 6:
        return np.nan
    return 1 - abs(spearmanr(pts[ok, 2], p[ok])[0])


def shot(args):
    path, variants = args
    row = load_row(path)
    Rg = np.asarray(row["efit_grid_R"], float); Zg = np.asarray(row["efit_grid_Z"], float)
    t = np.asarray(row["efit_times"], float)
    stem = Path(path).stem
    preds = {}
    for v in variants:
        f = BENCH / v / f"{stem}.npz"
        if not f.exists():
            return None
        preds[v] = np.load(f)
    frames = preds[variants[0]]["frames"]
    res = {v: [] for v in ("truth",) + tuple(variants)}
    for i, k in enumerate(frames):
        pts = thomson_points(row, t[k])
        res["truth"].append(frame_err(np.asarray(row["efit_psirz"][k], float), Rg, Zg, pts))
        for v in variants:
            res[v].append(frame_err(preds[v]["psirz"][i].astype(float), Rg, Zg, pts))
    return {v: np.nanmean(x) if np.isfinite(x).any() else np.nan for v, x in res.items()}


if __name__ == "__main__":
    files = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    variants = tuple(sys.argv[2:])
    with Pool(12) as pool:
        rows = [r for r in pool.map(shot, [(f, variants) for f in files]) if r]
    keys = ("truth",) + variants
    M = np.array([[r[k] for k in keys] for r in rows])
    M = M[np.isfinite(M).all(1)]
    rng = np.random.default_rng(0)
    print(f"shots {len(M)}")
    for j, k in enumerate(keys):
        print(f"{k:8s} err {M[:, j].mean():.4f}")
    base = M[:, 1]
    for j, k in enumerate(keys[2:], 2):
        d = M[:, j] - base
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"{k} - {keys[1]}: d_err {d.mean():+.4f} 95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
