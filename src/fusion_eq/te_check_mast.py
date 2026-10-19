"""te_check on MAST public preds (python -m fusion_eq.te_check_mast RUNS_DIR SHOTS_FILE): Te along the midplane core chord must be monotone in psi."""
import pickle, sys
from pathlib import Path
import numpy as np
from .paths import use_starter
from .te_check import frame_err

use_starter()
from data_fixes import align_mast_thomson_core  # noqa: E402


def shot_errs(row, psis: dict, every: int = 4) -> dict:
    """Mean te_check error per cache over every `every`-th frame with a Thomson pulse within 8 ms
    (midplane core chord at Z = 0); NaN where no frame qualifies."""
    Rg = np.asarray(row["efit_grid_R"], float); Zg = np.asarray(row["efit_grid_Z"], float)
    t = np.asarray(row["efit_times"], float)
    Rc, Te, _ = align_mast_thomson_core(row); tt = np.asarray(row["thomson_core_times"], float)
    e = {c: [] for c in psis}
    for k in range(0, len(t), every):
        j = np.abs(tt - t[k]).argmin()
        if abs(tt[j] - t[k]) > 8:
            continue
        m = np.isfinite(Te[j]) & (Te[j] > 0)
        pts = np.c_[Rc[m], np.zeros(m.sum()), Te[j][m]]
        for c in psis:
            e[c].append(frame_err(psis[c][k].astype(float), Rg, Zg, pts))
    return {c: np.nanmean(v) if np.isfinite(v).any() else np.nan for c, v in e.items()}


if __name__ == "__main__":
    runs = Path(sys.argv[1]); shots = [int(x) for x in open(sys.argv[2])]
    caches = {c: runs / p for c, p in (("c8", "preds_c8"), ("kp8", "mast/preds_kp8"), ("kp4", "mast/preds_kp4"), ("kp16", "mast/preds_kp16"), ("kp2", "mast/preds_kp2"), ("kp2v", "mast/preds_kp2v"),
                                       ("kp1", "mast/preds_kp1"), ("kp0", "mast/preds_kp0"))}
    M = []
    for s in shots:
        row = pickle.load(open(runs / f"mast/rows/{s:04d}.pkl", "rb"))
        e = shot_errs(row, {c: np.load(d / f"shot_{s:04d}.npz")["psirz"] for c, d in caches.items()})
        M.append([e[c] for c in caches])
    M = np.array(M); M = M[np.isfinite(M).all(1)]
    rng = np.random.default_rng(0)
    print("shots", len(M))
    for j, c in enumerate(caches):
        d = M[:, j] - M[:, 0]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"{c:5s} err {M[:, j].mean():.4f}  vs c8 {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
