"""Label-free checks of a MAST prediction cache against the shot's own Thomson profiles.

No MAST equilibrium enters here. Te and ne are flux functions, so on a correct flux map the
midplane Thomson channels inboard and outboard of the magnetic axis must give the same Te at the
same normalised flux (`te_iso`, `ne_iso`, rms mismatch over the profile's peak value). The
density edges should sit on psi_N = 1 (`edge_in`, `edge_out`, signed psi_N - 1). Whether a lower
number here means a higher leaderboard score was tested against the Development-phase scores and
failed: on the server's worst shots for c6 and c7sv these checks rated them better than c4
(docs/BENCH.md). Diagnostic only; nothing is selected with them.

    PYTHONPATH=src uv run python -m fusion_eq.mast_bench --cache DIR --out FILE.npz [--workers 12]
"""
import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .paths import RUNS_DIR

ROWS = RUNS_DIR / "mast" / "rows"
METRICS = ("te_iso", "ne_iso", "edge_in", "edge_out", "r_axis", "z_axis")
STRIDE = 3  # every 3rd EFIT frame: 15 ms apart, still ~23 frames per shot


def _psi_n(psi, R, Z, mc):
    """(psi_N on the Z = 0 row, R_axis, Z_axis), or None when no closed LCFS is found."""
    from derive import magnetic_axis
    from lcfs import extract_lcfs
    from scipy.interpolate import RegularGridInterpolator
    Ra, Za, iz, ir = magnetic_axis(psi, R, Z, "MAST", mc)
    c = extract_lcfs(psi, R, Z, "MAST", mc, mc.astype(np.float64), n_iter=14, n_points=256)
    if c is None or not np.isfinite(Ra):
        return None
    f = RegularGridInterpolator((Z, R), psi, bounds_error=False, fill_value=np.nan)
    p_b = np.nanmean(f(c[:, ::-1]))
    p_a = f([[Za, Ra]])[0]
    mid = f(np.c_[np.zeros_like(R), R])
    return (mid - p_a) / (p_b - p_a), Ra, Za


def _iso(x_in, y_in, x_out, y_out, scale):
    """rms of y_in - y_out(x_in) over inboard points inside the outboard psi_N range, / scale."""
    o = np.argsort(x_out)
    x_out, y_out = x_out[o], y_out[o]
    keep = (x_in > x_out[0]) & (x_in < x_out[-1])
    if keep.sum() < 3:
        return np.nan
    return float(np.sqrt(np.mean((y_in[keep] - np.interp(x_in[keep], x_out, y_out)) ** 2)) / scale)


def shot_metrics(row, pred) -> np.ndarray:
    """(n_frames, len(METRICS)) per scored frame; NaN where a check cannot be made."""
    from data_fixes import align_mast_thomson_core
    from .mast import _median5, machine_mask, thomson_edges
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    t = np.asarray(row["efit_times"], float)
    mc = machine_mask(row, R, Z)
    Rc, Te, ne = align_mast_thomson_core(row)
    o = np.argsort(Rc)
    Rc, Te, ne = Rc[o], _median5(np.asarray(Te, float)[:, o]), _median5(np.asarray(ne, float)[:, o])
    tt = np.asarray(row["thomson_core_times"], float)
    rin, rout = thomson_edges(row, t)
    frames = np.arange(0, len(t), STRIDE)
    out = np.full((len(frames), len(METRICS)), np.nan)
    for j, k in enumerate(frames):
        r = _psi_n(np.asarray(pred["psirz"][k], float), R, Z, mc)
        if r is None:
            continue
        pn_grid, Ra, Za = r
        out[j, 4:] = Ra, Za
        pn = np.interp(Rc, R, pn_grid)
        for m, e in enumerate((rin[k], rout[k])):
            if np.isfinite(e):
                out[j, 2 + m] = np.interp(e, R, pn_grid) - 1
        i = int(np.abs(tt - t[k]).argmin())
        for m, prof in enumerate((Te[i], ne[i])):
            ok = np.isfinite(prof) & (prof > 0) & np.isfinite(pn) & (pn > 0.05) & (pn < 0.9)
            a, b = ok & (Rc < Ra), ok & (Rc > Ra)
            if a.sum() >= 3 and b.sum() >= 3:
                out[j, m] = _iso(pn[a], prof[a], pn[b], prof[b], np.nanmax(prof[ok]))
    return out


def _init():
    from .paths import use_starter
    use_starter()
    import local_score  # noqa: F401  puts fusion_scoring/ on sys.path


def _work(args):
    i, cache = args
    with open(ROWS / f"{i:04d}.pkl", "rb") as fh:
        row = pickle.load(fh)
    return i, shot_metrics(row, np.load(Path(cache) / f"shot_{i:04d}.npz"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-shots", type=int, default=0)
    a = ap.parse_args()
    shots = sorted(int(p.stem[5:]) for p in Path(a.cache).glob("shot_[0-9][0-9][0-9][0-9].npz"))
    shots = shots[:a.max_shots] if a.max_shots else shots
    res = {}
    with Pool(a.workers, initializer=_init) as pool:
        for i, m in pool.imap_unordered(_work, [(i, a.cache) for i in shots], chunksize=4):
            res[i] = m
    idx = sorted(res)
    np.savez(a.out, shots=np.array(idx), metrics=np.array(METRICS),
             **{f"s{i:04d}": res[i] for i in idx})
    allm = np.concatenate([res[i] for i in idx])
    print(a.cache, {k: round(float(np.nanmedian(np.abs(allm[:, n]))), 4) for n, k in enumerate(METRICS)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
