"""Current-profile closure read off a labelled equilibrium.

freeb solves with J = (beta0 R/r0 + (1 - beta0) r0/R) (1 - psi_N)^alpha and generic alpha 1. Here
(alpha, beta0) are fitted per EFIT frame to J_phi = -Delta* psi / (mu0 R) inside the LCFS: the
target a closure model learns on DIII-D, and the per-frame oracle (d3d_bench `orc` variants) that
bounds what any closure in this family can add.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ALPHAS = np.round(np.arange(0.2, 4.01, 0.1), 2)
PN_MAX = 0.95  # leave out the last flux surfaces, where the finite difference crosses the LCFS kink


def delta_star(psi, R, Z):
    """Delta* psi = d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2 on the (nZ, nR) grid."""
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    pR = np.gradient(psi, dR, axis=1)
    return (np.gradient(pR, dR, axis=1) - pR / R[None, :]
            + np.gradient(np.gradient(psi, dZ, axis=0), dZ, axis=0))


def fit_frame(psi, R, Z, lcfs, axis):
    """(alpha, beta0, rel_rms) of the freeb family fitted to this frame's current, or NaNs.
    `lcfs` (n, 2) boundary points, `axis` (R, Z) of the magnetic axis."""
    from derive import _bilinear, _points_in_polygon
    Rg, Zg = np.meshgrid(R, Z)
    inside = _points_in_polygon(Rg.ravel(), Zg.ravel(), lcfs).reshape(Rg.shape)
    if inside.sum() < 30:
        return np.nan, np.nan, np.nan
    p_ax = float(_bilinear(psi, R, Z, np.array([axis[0]]), np.array([axis[1]]))[0])
    p_b = float(np.mean(_bilinear(psi, R, Z, lcfs[:, 0], lcfs[:, 1])))
    if p_ax == p_b:
        return np.nan, np.nan, np.nan
    pn = np.clip((psi - p_ax) / (p_b - p_ax), 0, 1)
    use = inside & (pn < PN_MAX)
    j = delta_star(psi, R, Z)[use] / Rg[use]
    j = j * np.sign(j.sum())  # current along Ip: either stored sign
    r0 = float(Rg[inside].mean())
    best = (np.inf, np.nan, np.nan)
    for a in ALPHAS:
        s = (1 - pn[use]) ** a
        A = np.stack([Rg[use] / r0 * s, r0 / Rg[use] * s], 1)
        c, res = nnls(A, j)
        if res < best[0] and c.sum() > 0:
            best = (res, a, c[0] / c.sum())
    return best[1], best[2], best[0] / np.linalg.norm(j)


def fit_row(row, frames):
    """(len(frames), 3) array of fit_frame on a DIII-D parquet row's EFIT frames."""
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    out = np.full((len(frames), 3), np.nan)
    for j, k in enumerate(frames):
        n = int(np.asarray(row["efit_lcfs_n"])[k])
        if n < 5:
            continue
        lcfs = np.stack([np.asarray(row["efit_lcfs_r"][k], float)[:n],
                         np.asarray(row["efit_lcfs_z"][k], float)[:n]], 1)
        axis = (float(np.asarray(row["efit_r_axis"])[k]), float(np.asarray(row["efit_z_axis"])[k]))
        if np.isfinite(axis).all():
            out[j] = fit_frame(np.asarray(row["efit_psirz"][k], float), R, Z, lcfs, axis)
    return out


# Learned closure: alpha from inputs every machine records (Thomson Te/ne, Ip, TF current), no flux label.
MU0 = 4e-7 * np.pi
E_CHARGE = 1.602176634e-19
KAPPA = 1.8  # the solver's elongation prior (mast.shaped_current); no per-frame elongation exists at test time
BINS = np.array([0, .3, .5, .7, .85, 1.0])  # psi_N bins of the pressure average in beta_p
# Profile shape as inner log-gradients over psi_N, not ratios of fixed bins: DIII-D's core chord passes
# off axis (psi_N > .15 on EFIT, > .3 on a solve with a misplaced axis), MAST's midplane chord crosses it.
# No outer gradient: on the bench it does not carry over from EFIT psi to a solve (r .08 Te, .12 ne).
FEATURES = ("te_in", "ne_in", "log_qcyl", "log_betap", "log_t_tauR", "ramp_tauR")


def tau_r(a, te_kev):
    """Resistive time mu0 a^2 / eta, Spitzer eta = 2.8e-8 Te[keV]^-1.5 Ohm m (ln Lambda 17, Zeff 1; Wesson,
    Tokamaks, 4th ed., sec. 2.16), i.e. ~45 a^2 Te^1.5 s."""
    return MU0 * a ** 2 * te_kev ** 1.5 / 2.8e-8


def frame_features(psi, R, Z, rin, rout, pts, ip, dip, t_since, b0):
    """FEATURES of one frame. `psi` either sign (only its surfaces are used, to put Thomson in flux
    coordinates); `pts` (n, 4) Thomson R, Z, Te [eV], ne [m^-3]; ip [A]; dip [A/s]; t_since [s] since
    |Ip| first reached 10% of its peak; b0 [T] vacuum toroidal field at (rin + rout) / 2."""
    from derive import _bilinear
    out = np.full(len(FEATURES), np.nan)
    a, r0 = (rout - rin) / 2, (rout + rin) / 2
    if not (np.isfinite(a) and a > 0 and len(pts) >= 6 and np.isfinite(ip) and ip != 0):
        return out
    Rg, Zg = np.meshgrid(R, Z)
    box = (Rg > rin + 0.2 * a) & (Rg < rout - 0.2 * a) & (np.abs(Zg) < a)
    c = _bilinear(psi, R, Z, np.array([r0, rout]), np.array([0.0, 0.0]))
    i = np.argmax(np.where(box, np.sign(c[0] - c[1]) * psi, -np.inf))
    p_ax = psi.flat[i]
    p_b = float(_bilinear(psi, R, Z, np.array([rout]), np.array([Zg.flat[i]]))[0])
    if not np.isfinite(p_b) or p_b == p_ax:
        return out
    pn = (_bilinear(psi, R, Z, pts[:, 0], pts[:, 1]) - p_ax) / (p_b - p_ax)
    te, ne = pts[:, 2], pts[:, 3]
    inside = (pn >= 0) & (pn < 1)
    if inside.sum() < 6:
        return out
    k = np.digitize(pn, BINS) - 1
    p = [np.median(2 * E_CHARGE * te[k == b] * ne[k == b]) for b in range(len(BINS) - 1) if (k == b).any()]
    tau = tau_r(a, np.percentile(te[inside], 90) / 1e3)
    bp = MU0 * abs(ip) / (2 * np.pi * a * np.sqrt((1 + KAPPA ** 2) / 2))
    out[:] = (_slope(pn, te, 0, .6), _slope(pn, ne, 0, .6),
              np.log(5 * a ** 2 * b0 * (1 + KAPPA ** 2) / 2 / (r0 * abs(ip) / 1e6)),
              np.log(2 * MU0 * np.mean(p) / bp ** 2),
              np.log(max(t_since, 1e-3) / tau), dip / ip * tau)
    return out


def _slope(pn, v, lo, hi):
    """Theil-Sen d log v / d psi_N over lo <= psi_N < hi; NaN under 4 points or a psi_N span under .15."""
    from scipy.stats import theilslopes
    m = (pn >= lo) & (pn < hi)
    if m.sum() < 4 or np.ptp(pn[m]) < 0.15:
        return np.nan
    return theilslopes(np.log(v[m]), pn[m])[0]


def ip_history(t_ip, ip, t, dt):
    """(ip, dip/dt, t_since) at times t on the Ip series (t_ip, ip), same units; dt the derivative step."""
    from .greens import interp_signal
    x = interp_signal(t_ip, ip, t)
    d = (interp_signal(t_ip, ip, t + dt) - interp_signal(t_ip, ip, t - dt)) / (2 * dt)
    a = np.abs(np.asarray(ip, float))
    on = np.asarray(t_ip, float)[np.argmax(a >= 0.1 * np.nanmax(a))]
    return x, d, t - on


def d3d_thomson(row, t, dt=8.0):
    """(n, 4) R, Z, Te, ne of DIII-D core + tangential chords at the pulse nearest t (ms)."""
    names = [str(x) for x in row["thomson_chord_name"]]
    R = np.asarray(row["thomson_chord_R"], float); Z = np.asarray(row["thomson_chord_Z"], float)
    out = []
    for kind, key in (("core", "thomson_core"), ("tangential", "thomson_edge")):
        idx = [i for i, n in enumerate(names) if f"_{kind}_" in n]
        tt = np.asarray(row[key + "_times"], float)
        Te = np.asarray(row[key + "_Te"], float); ne = np.asarray(row[key + "_ne"], float)
        good = np.where(((Te > 0) & (ne > 0)).sum(1) >= 3)[0]
        if not len(good):
            continue
        j = good[np.abs(tt[good] - t).argmin()]
        if abs(tt[j] - t) > dt:
            continue
        m = (Te[j] > 0) & (ne[j] > 0)
        out.append(np.c_[R[idx][m], Z[idx][m], Te[j][m], ne[j][m]])
    return np.vstack(out) if out else np.zeros((0, 4))


def d3d_features(row, frames, psis):
    """(len(frames), len(FEATURES)) on DIII-D, Thomson mapped on `psis` (EFIT for training, a solve on the bench)."""
    from data_fixes import fix_d3d_ip_times
    from .d3d_bench import N_TF, edges
    from .greens import interp_signal
    from .qprofile import vacuum_f
    R = np.asarray(row["efit_grid_R"], float); Z = np.asarray(row["efit_grid_Z"], float)
    t = np.asarray(row["efit_times"], float)[frames]  # ms
    ip, dip, ts = ip_history(fix_d3d_ip_times(row), np.asarray(row["magnetics_plasma_current"], float) * 1e3, t, 5.0)
    f = vacuum_f(interp_signal(row["magnetics_time"], row["magnetics_bcoil"], t) * 1e3, N_TF)
    X = np.full((len(frames), len(FEATURES)), np.nan)
    for j, k in enumerate(frames):
        rin, rout = edges(row, k)
        X[j] = frame_features(np.asarray(psis[j], float), R, Z, rin, rout, d3d_thomson(row, t[j]),
                              ip[j], dip[j] * 1e3, ts[j] / 1e3, f[j] / ((rin + rout) / 2))
    return X


def mast_features(row, psis, dt=8.0):
    """(T, len(FEATURES)) on a MAST row (times in ms, as on DIII-D): midplane core Thomson mapped on
    `psis`, a solve of the same row; edges from Thomson density as in predict_mast."""
    from data_fixes import align_mast_thomson_core
    from .greens import interp_signal
    from .mast import N_TF, thomson_edges
    from .qprofile import vacuum_f
    R = np.asarray(row["efit_grid_R"], float); Z = np.asarray(row["efit_grid_Z"], float)
    t = np.asarray(row["efit_times"], float)
    rin, rout = thomson_edges(row, t)
    ip, dip, ts = ip_history(row["magnetics_time"], np.asarray(row["magnetics_plasma_current"], float) * 1e3, t, 5.0)
    f = vacuum_f(interp_signal(row["magnetics_time"], row["magnetics_tf_current"], t) * 1e3, N_TF)
    Rc, Te, ne = align_mast_thomson_core(row)
    Te, ne = np.asarray(Te, float), np.asarray(ne, float)
    tt = np.asarray(row["thomson_core_times"], float)
    good = np.where(((Te > 0) & (ne > 0)).sum(1) >= 3)[0]
    X = np.full((len(t), len(FEATURES)), np.nan)
    for k in range(len(t) if len(good) else 0):
        j = good[np.abs(tt[good] - t[k]).argmin()]
        if abs(tt[j] - t[k]) > dt:
            continue
        m = (Te[j] > 0) & (ne[j] > 0)
        X[k] = frame_features(np.asarray(psis[k], float), R, Z, rin[k], rout[k],
                              np.c_[Rc[m], np.zeros(m.sum()), Te[j][m], ne[j][m]],
                              ip[k], dip[k] * 1e3, ts[k] / 1e3, f[k] / ((rin[k] + rout[k]) / 2))
    return X


def predict_alpha(model, X):
    """(alpha, spread) per row of FEATURES from a `fit` pickle (path or dict): geometric mean over the
    bagged models on inputs clipped to the training [p1, p99], spread the sd of their log alpha.
    Rows with a missing feature keep the generic alpha 1."""
    import pickle
    if not isinstance(model, dict):
        with open(model, "rb") as fh:
            model = pickle.load(fh)
    X = np.asarray(X, float)
    ok = np.isfinite(X).all(1)
    a, s = np.ones(len(X)), np.full(len(X), np.nan)
    if ok.any():
        la = np.array([g.predict(np.clip(X[ok], model["lo"], model["hi"])) for g in model["models"]])
        a[ok] = np.clip(np.exp(la.mean(0)), ALPHAS[0], ALPHAS[-1])
        s[ok] = la.std(0)
    return a, s


def mast_alpha(row, psis, model):
    """Per-frame alpha for predict_mast from the closure `model`, features read off the solve `psis`."""
    return predict_alpha(model, mast_features(row, psis))[0]


def _data_shot(path, n_frames=10):
    from .d3d_bench import load_row
    row = load_row(path)
    T = len(row["efit_times"])
    frames = np.unique(np.linspace(0, T - 1, min(T, n_frames)).astype(int))
    psis = [np.asarray(row["efit_psirz"][k], float) for k in frames]
    y = fit_row(row, frames)
    li = np.asarray(row["efit_li"], float)[frames]
    return d3d_features(row, frames, psis), y, li


def fit(data, out, n_bag=5, max_rms=0.3):
    """log alpha* from FEATURES by gradient boosting on the frames the freeb family fits (rel_rms <
    max_rms), bagged over shots. Prints the shot-grouped 5-fold held-out skill next to a ridge reference."""
    import json
    import pickle
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = np.load(data)
    ok = np.isfinite(d["X"]).all(1) & np.isfinite(d["y"][:, 0]) & (d["y"][:, 2] < max_rms)
    X, la, li, shot = d["X"][ok], np.log(d["y"][ok, 0]), d["li"][ok], d["shot"][ok]
    lo, hi = np.percentile(X, [1, 99], axis=0)
    X = np.clip(X, lo, hi)

    def gbm(seed):
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=50, l2_regularization=1.0, random_state=seed)
    cv = {"gbm": np.zeros(len(la)), "ridge": np.zeros(len(la))}
    for tr, te in GroupKFold(5).split(X, la, shot):
        cv["gbm"][te] = gbm(0).fit(X[tr], la[tr]).predict(X[te])
        cv["ridge"][te] = make_pipeline(StandardScaler(), RidgeCV()).fit(X[tr], la[tr]).predict(X[te])
    fl = np.isfinite(li)
    rep = {"frames": int(ok.sum()), "of": len(ok), "shots": len(np.unique(shot)),
           "r_li_target": float(np.corrcoef(la[fl], li[fl])[0, 1])}
    for k, p in cv.items():
        rep[k] = {"r": float(np.corrcoef(p, la)[0, 1]),
                  "r2": float(1 - ((p - la) ** 2).sum() / ((la - la.mean()) ** 2).sum()),
                  "r_li": float(np.corrcoef(p[fl], li[fl])[0, 1])}
    rng = np.random.default_rng(0)
    ids = np.unique(shot)
    models = []
    for b in range(n_bag):
        idx = np.concatenate([np.flatnonzero(shot == s) for s in rng.choice(ids, len(ids))])
        models.append(gbm(b).fit(X[idx], la[idx]))
    with open(out, "wb") as fh:
        pickle.dump({"models": models, "lo": lo, "hi": hi, "features": FEATURES, "cv": rep}, fh)
    print(json.dumps(rep, indent=1))


def main() -> int:
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(description="learned current closure on DIII-D EFIT frames")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("data", help="features and fitted (alpha, beta0, rel_rms) per EFIT frame")
    g.add_argument("--files", required=True, help="text file of parquet paths")
    g.add_argument("--exclude", help="text file of parquet paths to leave out (the bench)")
    g.add_argument("--n", type=int, default=1500)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--out", required=True)
    f = sub.add_parser("fit", help="closure model pickle from a `data` file")
    f.add_argument("--data", required=True)
    f.add_argument("--model", required=True)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.data, a.model)
        return 0
    files = [x.strip() for x in open(a.files) if x.strip()]
    if a.exclude:
        skip = {Path(x.strip()).name for x in open(a.exclude) if x.strip()}
        files = [f for f in files if Path(f).name not in skip]
    files = sorted(np.random.default_rng(0).permutation(files)[:a.n])
    from .d3d_bench import _init
    with Pool(a.workers, initializer=_init) as p:
        res = p.map(_data_shot, files, chunksize=4)
    shot = np.concatenate([np.full(len(r[1]), i) for i, r in enumerate(res)])
    np.savez(a.out, X=np.vstack([r[0] for r in res]), y=np.vstack([r[1] for r in res]),
             li=np.concatenate([r[2] for r in res]), shot=shot, files=np.array(files))
    print(f"{len(files)} shots {len(shot)} frames -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
