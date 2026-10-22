"""Physics-taught twin: a residual model on top of the physics solve, trained only on synthetic
equilibria from our own free-boundary solver. No equilibrium label enters training.

For each real frame (its coil currents, Ip and edge radii, all inputs) we draw a "true"
equilibrium whose unknowns are randomised: current profile (1 - psi_N^a1)^a2 and pressure share
beta0, start shape, coil-current calibration errors, and a uniform vertical / radial error field
of the kind vessel currents leave. The edges the pipeline would see are read off that truth (with
Thomson-like error on MAST), and the shipped pipeline (4 Picard passes, beta0 matched to the outer
edge) is run on the *unperturbed* coils. The model maps (pipeline psi, coil flux, edges, Ip,
force-balance z) to the truth minus the pipeline psi. Since R^2 rewards the conditional mean, it
learns the correction the pipeline needs on average over what the inputs leave undetermined.

The generator is the same for DIII-D and MAST; only the machine inputs differ. DIII-D is the
transfer test: train on synthetic DIII-D from the bench shots' inputs, score on EFIT truth
(d3d_bench variant "twin").

    PYTHONPATH=src uv run python -m fusion_eq.twin gen --machine d3d --files LIST --out DIR
    PYTHONPATH=src uv run python -m fusion_eq.twin fit --data DIR --model M.pkl
"""
import argparse
import pickle
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .paths import npz_done, save_npz

COIL_ERR = 0.03  # relative sd per coil column
FIELD_ERR = (0.02, 0.005)  # sd of uniform error Bz, Br (T)
EDGE_ERR = (0.03, 0.02)  # MAST Thomson edge vs LCFS: bias inward, sd (m); as d3d_bench.EDGE_ERR
EDGE_REJECT = 0.15  # drop draws whose edges land this far (m) from the real frame's
K_PHI, K_VAC = 32, 16


def d3d_inputs(row, frames):
    """(R, Z, G (nC, nZ, nR), amps (T, nC) in solver sign, ip (T,) > 0, rin, rout, mask)."""
    from data_fixes import fix_d3d_ip_times
    from .d3d_bench import E_SCALE, edges
    from .greens import coil_greens, interp_signal
    from .mast import machine_mask
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    t = np.asarray(row["efit_times"], float)[frames]
    gs = coil_greens(row, R, Z)
    amps = np.stack([interp_signal(row["magnetics_time"], row[c], t) * 1e3
                     * (E_SCALE if c == "magnetics_ECOILA" else 1.0) for c in gs], axis=1)
    ip = interp_signal(fix_d3d_ip_times(row), row["magnetics_plasma_current"], t) * 1e3
    s = np.sign(np.nanmedian(ip)) or 1.0
    rin, rout = np.array([edges(row, k) for k in frames]).T
    return R, Z, np.stack(list(gs.values())), s * amps, np.abs(ip), rin, rout, machine_mask(row, R, Z)


def mast_inputs(row, frames, sol_scale):
    from .greens import coil_greens, interp_signal
    from .mast import machine_mask, thomson_edges
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    t = np.asarray(row["efit_times"], float)[frames]
    gs = coil_greens(row, R, Z)
    amps = np.stack([interp_signal(row["magnetics_time"], row[c], t) * 1e3
                     * (sol_scale if c == "magnetics_sol_current" else 1.0) for c in gs], axis=1)
    ip = interp_signal(row["magnetics_time"], row["magnetics_plasma_current"], t) * 1e3
    rin, rout = thomson_edges(row, t)
    return R, Z, np.stack(list(gs.values())), amps, ip, rin, rout, machine_mask(row, R, Z)


def pipeline(vac, R, Z, ip, mc, rin, rout):
    """The shipped solve: Thomson-centred D shape, beta0 matched to the outer edge, 4 passes."""
    from .freeb import solve_matched
    from .mast import shaped_current
    J = shaped_current(R, Z, (rin + rout) / 2, (rout - rin) / 2, 1.8, 0.4)
    return solve_matched(vac, J, R, Z, ip, mc, mc.astype(np.float64), rout, n_iter=4)[0]


def force_z(psi, vac, R, Z, mc):
    """Force-balance vertical shift of the solved current (mast_vertical, one frame, no median)."""
    from .mast_vertical import Z_MAX, force_zero, plasma_current
    dZ = Z[1] - Z[0]
    z = force_zero(plasma_current(psi - vac, R, Z, mc), np.gradient(vac, dZ, axis=0), dZ)
    return float(np.clip(np.nan_to_num(z), -Z_MAX, Z_MAX))


def synth(G, amps, ip, R, Z, mc, rin, rout, rng, edge_err):
    """One draw: (phi_true, phi_pipeline, scalars) with phi = (psi - observed coil flux) / ip, or
    None when the truth has no closed LCFS or lands far from the real frame's edges."""
    from lcfs import extract_lcfs
    from .freeb import solve
    from .mast import shaped_current
    Rg, Zg = np.meshgrid(R, Z)
    vac = np.tensordot(amps, G, 1)
    eps = rng.normal(0, COIL_ERR, len(amps))
    vt = (np.tensordot(amps * (1 + eps), G, 1) + rng.normal(0, FIELD_ERR[0]) * Rg ** 2 / 2
          - rng.normal(0, FIELD_ERR[1]) * Rg * Zg)
    a1, a2, beta0 = rng.uniform(1, 3), rng.uniform(0.5, 3), rng.uniform(0, 0.8)
    J0 = shaped_current(R, Z, (rin + rout) / 2 + rng.normal(0, 0.03),
                        (rout - rin) / 2 * rng.uniform(0.85, 1.1), rng.uniform(1.4, 2.2),
                        rng.uniform(0, 0.6), rng.normal(0, 0.03), alpha=a2)
    psi_t = solve(vt, J0, R, Z, ip, mc, mc.astype(np.float64), beta0=beta0, n_iter=16,
                  profile=lambda pn, inside: (1 - pn ** a1) ** a2, vertical=rng.random() < 0.5)[0]
    c = extract_lcfs(psi_t, R, Z, "MAST", mc, mc.astype(np.float64), n_iter=14, n_points=256)
    if c is None:
        return None
    ri, ro = c[:, 0].min(), c[:, 0].max()
    if edge_err:
        ri += EDGE_ERR[0] + EDGE_ERR[1] * rng.standard_normal()
        ro -= EDGE_ERR[0] - EDGE_ERR[1] * rng.standard_normal()
    if max(abs(ri - rin), abs(ro - rout)) > EDGE_REJECT or ro <= ri:
        return None
    psi_h = pipeline(vac, R, Z, ip, mc, ri, ro)
    scal = np.array([ri, ro, ip / 1e6, force_z(psi_h, vac, R, Z, mc)])
    return (psi_t - vac) / ip, (psi_h - vac) / ip, vac / ip, scal


def _gen(args):
    machine, src, out, n_frames, draws, sol_scale = args
    dst = Path(out) / (Path(src).stem + ".npz")
    if npz_done(dst):
        return
    if machine == "d3d":
        from .d3d_bench import frames_of, load_row
        row = load_row(src)
        cand = frames_of(len(row["efit_times"]))
    else:
        row = pickle.load(open(src, "rb"))
        cand = np.arange(len(row["efit_times"]))
    rng = np.random.default_rng(zlib.crc32(Path(src).stem.encode()))
    frames = np.sort(rng.choice(cand, min(n_frames, len(cand)), replace=False))
    R, Z, G, amps, ip, rin, rout, mc = (d3d_inputs(row, frames) if machine == "d3d"
                                        else mast_inputs(row, frames, sol_scale(src)))
    rows = []
    for j in range(len(frames)):
        if not (np.isfinite(rin[j]) and rout[j] > rin[j] and ip[j] > 1e4):
            continue
        for _ in range(draws):
            r = synth(G, amps[j], ip[j], R, Z, mc, rin[j], rout[j], rng, machine == "mast")
            if r is not None:
                rows.append(r)
    if not rows:
        save_npz(dst, phi_t=np.zeros((0, R.size * Z.size), np.float32))
        return
    pt, ph, vn, sc = (np.stack(x) for x in zip(*rows))
    save_npz(dst, phi_t=pt.reshape(len(pt), -1).astype(np.float32),
             phi_h=ph.reshape(len(ph), -1).astype(np.float32),
             vacn=vn.reshape(len(vn), -1).astype(np.float32), scal=sc.astype(np.float32))


class Twin:
    """PCA bases for phi and coil flux, a standardised regressor on the coefficients."""

    def __init__(self, phi_basis, vac_basis, xm, xs, reg):
        self.phi_basis, self.vac_basis, self.xm, self.xs, self.reg = phi_basis, vac_basis, xm, xs, reg

    def x(self, phi_h, vacn, scal):
        c = lambda a, B: (a - a.mean(1, keepdims=True)) @ B.T
        return (np.hstack([c(phi_h, self.phi_basis), c(vacn, self.vac_basis), scal]) - self.xm) / self.xs

    def delta(self, phi_h, vacn, scal):
        """(n, nZ*nR) correction to phi, zero mean over the grid (an offset moves no contour)."""
        return self.reg.predict(self.x(phi_h, vacn, scal)) @ self.phi_basis


def _pca(a, k):
    a = a - a.mean(1, keepdims=True)
    return np.linalg.svd(a[:: max(1, len(a) // 8000)], full_matrices=False)[2][:k]


def load_data(data):
    fs = sorted(Path(data).glob("*.npz"))
    d = [np.load(f) for f in fs]
    d = [(x, i) for i, x in enumerate(d) if len(x["phi_t"])]
    cat = lambda k: np.concatenate([x[k] for x, _ in d]).astype(np.float64)
    group = np.concatenate([np.full(len(x["phi_t"]), i) for x, i in d])
    return cat("phi_t"), cat("phi_h"), cat("vacn"), cat("scal"), group


def fit(data, model_out, kind="mlp", seed=0):
    """Fit on 80% of shots, report held-out gain on synthetic, refit on all, save."""
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    pt, ph, vn, sc, g = load_data(data)
    pb = _pca(np.vstack([pt, ph]), K_PHI)
    vb = _pca(vn, K_VAC)
    tw = Twin(pb, vb, 0.0, 1.0, None)
    X = tw.x(ph, vn, sc)
    tw.xm, tw.xs = X.mean(0), X.std(0) + 1e-12
    X = (X - tw.xm) / tw.xs
    r = pt - ph
    y = (r - r.mean(1, keepdims=True)) @ pb.T

    def make():
        return (Ridge(alpha=1.0) if kind == "ridge" else
                MLPRegressor(hidden_layer_sizes=(256, 256), alpha=1e-4, early_stopping=True, max_iter=300, random_state=seed))
    rng = np.random.default_rng(seed)
    shots = np.unique(g)
    val = np.isin(g, rng.choice(shots, len(shots) // 5, replace=False))
    tw.reg = make().fit(X[~val], y[~val])
    res0 = r[val] - r[val].mean(1, keepdims=True)
    res1 = res0 - tw.reg.predict(X[val]) @ pb
    print(f"{kind}: n {len(X)} shots {len(shots)}  held-out residual variance kept "
          f"{(res1 ** 2).sum() / (res0 ** 2).sum():.3f}  (1 = no gain)")
    tw.reg = make().fit(X, y)
    pickle.dump(tw, open(model_out, "wb"))
    return tw


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--machine", choices=("d3d", "mast"), required=True)
    g.add_argument("--files", help="d3d: text file of parquet paths")
    g.add_argument("--rows", help="mast: directory of row pickles")
    g.add_argument("--sol-from", help="mast: cache whose per-shot sol_scale is used")
    g.add_argument("--out", required=True)
    g.add_argument("--frames", type=int, default=6)
    g.add_argument("--draws", type=int, default=3)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--workers", type=int, default=4)
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--model", required=True)
    f.add_argument("--kind", choices=("mlp", "ridge"), default="mlp")
    p = sub.add_parser("apply", help="mast: twin-corrected copy of a per-shot cache")
    p.add_argument("--cache", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--c", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from .d3d_bench import _init
    _init()
    if a.cmd == "fit":  # through the package so the pickle names fusion_eq.twin.Twin, not __main__
        from fusion_eq import twin
        twin.fit(a.data, a.model, a.kind)
        return 0
    if a.cmd == "apply":
        from fusion_eq import twin  # the pickled Twin lives in the package module
        Path(a.out).mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), a.rows, a.out, a.model, a.c) for f in sorted(Path(a.cache).glob("shot_*.npz"))]
        with Pool(a.workers, initializer=_init) as pool:
            for _ in pool.imap_unordered(twin._apply, jobs):
                pass
        print("applied", len(list(Path(a.out).glob("shot_*.npz"))), "shots")
        return 0
    srcs = ([l.strip() for l in open(a.files) if l.strip()] if a.machine == "d3d"
            else [str(p) for p in sorted(Path(a.rows).glob("*.pkl"))])
    srcs = srcs[: a.limit] if a.limit else srcs
    sol = _SolFrom(a.sol_from)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with Pool(a.workers, initializer=_init) as pool:
        for _ in pool.imap_unordered(_gen, [(a.machine, s, a.out, a.frames, a.draws, sol) for s in srcs]):
            pass
    print(a.machine, len(list(Path(a.out).glob("*.npz"))), "shots")
    return 0


def mast_apply(row, d: dict, tw, c: float) -> dict:
    """Twin-corrected copy of a MAST cache entry (iter4: psi already time-smoothed by finalize).
    The correction is smoothed over the same SMOOTH frames, q95 is re-measured on the new psi as
    finalize does, betaN is untouched. Constant fallbacks are returned as is."""
    from .greens import interp_signal
    from .mast import N_TF, SMOOTH, WALL_R, robust_series, smooth_frames
    from .qprofile import q_at, vacuum_f
    if str(d["method"]) == "const":
        return d
    frames = np.arange(len(row["efit_times"]))
    R, Z, G, amps, ip, rin, rout, mc = mast_inputs(row, frames, float(d["sol_scale"]))
    vac = np.tensordot(amps, G, 1)
    psi = d["psirz"].astype(np.float64)
    delta = np.zeros_like(psi)
    for k in np.flatnonzero(np.isfinite(rin) & (rout > rin) & (ip > 1e4)):
        phi = ((psi[k] - vac[k]) / ip[k]).reshape(1, -1)
        sc = np.array([[rin[k], rout[k], ip[k] / 1e6, force_z(psi[k], vac[k], R, Z, mc)]])
        delta[k] = ip[k] * tw.delta(phi, (vac[k] / ip[k]).reshape(1, -1), sc).reshape(psi[k].shape)
    psi = psi + c * smooth_frames(delta)
    t = np.asarray(row["efit_times"], float)
    f = vacuum_f(interp_signal(row["magnetics_time"], row["magnetics_tf_current"], t) * 1e3, N_TF)
    qn = robust_series(np.array([q_at(psi[k], R, Z, f[k], "MAST", mask_coarse=mc,
                                      mask_f=mc.astype(np.float64), wall_r=WALL_R)
                                 for k in range(len(t))]), 2, 25)
    q95 = smooth_frames(np.where(np.isfinite(qn), qn, d["q95"]), SMOOTH)
    return {**d, "psirz": psi.astype(np.float32), "q95": q95.astype(np.float32)}


def _apply(args):
    src, rows, out, model, c = args
    import os
    dst = Path(out) / Path(src).name
    if dst.exists():
        return
    row = pickle.load(open(Path(rows) / f"{Path(src).stem[5:]}.pkl", "rb"))
    d = mast_apply(row, dict(np.load(src)), pickle.load(open(model, "rb")), c)
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez(tmp, **d)
    os.replace(tmp, dst)


class _SolFrom:
    """Per-shot solenoid scale from a MAST cache (picklable for the pool)."""

    def __init__(self, cache):
        self.cache = cache

    def __call__(self, src):
        from .mast import SOL_SCALE
        if not self.cache:
            return SOL_SCALE
        p = Path(self.cache) / f"shot_{Path(src).stem}.npz"
        return float(np.load(p)["sol_scale"]) if p.exists() else SOL_SCALE


if __name__ == "__main__":
    raise SystemExit(main())
