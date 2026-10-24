"""Bench A: run the MAST free-boundary pipeline on labelled DIII-D shots and score it.

MAST has no labels we may use, so a MAST-side change (vertical force balance, current profile)
is judged here instead: same solver, same shape priors, DIII-D coils and EFIT truth. Three
substitutions keep it runnable, each applied identically to every variant so paired differences
stay fair:
  - the Thomson edges are replaced by the EFIT LCFS inner and outer radii (DIII-D's Thomson laser
    is vertical and gives no midplane profile);
  - the E-coil term is scaled by E_SCALE, the median least-squares fit on 8 training shots (the
    shipped convention lumps its turns into one conductor);
  - each frame gets the flux offset that best fits EFIT (no vessel-current model). An offset
    moves no contour, so the LCFS, the seven derived scalars and q95 are untouched.
q95 is measured on each variant's psi (qprofile.q_at, F from the TF current at N_TF turns, the
DIII-D design count; on EFIT psi this reproduces EFIT q95 within 0.5%), falling back to the
shot's EFIT mean where no surface is found. betaN is the shot's EFIT mean for every variant.

    PYTHONPATH=src uv run python -m fusion_eq.d3d_bench --files LIST --variant sym --out DIR
    PYTHONPATH=src uv run python -m fusion_eq.d3d_bench --files LIST --score sym vert --out DIR
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .paths import npz_done, save_npz

E_SCALE = 65.0
STRIDE = 10  # every 10th EFIT frame
# FUSION_BENCH_DENSE=N takes the first N consecutive EFIT frames instead (20 ms apart): the
# temporal-smoothing analogue needs neighbouring frames, as on MAST (5 ms apart)
DENSE = int(os.environ.get("FUSION_BENCH_DENSE", 0))


def frames_of(T):
    return np.arange(min(T, DENSE)) if DENSE else np.arange(0, T, STRIDE)
EDGE_ERR = (0.03, 0.02)  # (bias inward, sd) in m: Thomson 30% density edge vs LCFS, assumed
N_TF = 144  # DIII-D TF turns (design value; EFIT q95 / q95 at one turn gives 143.4)
VARIANTS = {
    "sym": {},
    "vert": {"vertical": True},
    "alpha2": {"alpha": 2.0},
    "alpha05": {"alpha": 0.5},
    "gap": {"dbz": (-0.04, -0.02, 0.0, 0.02, 0.04)},
    "iter16": {"n_iter": 16},
    "iter4": {"n_iter": 4},
    "iter6": {"n_iter": 6},
    # EDGE_ERR: edges as a density edge would give them, inside the LCFS and jittered
    "sym_e": {"edge_err": True},
    "iter4_e": {"n_iter": 4, "edge_err": True},
    "iter2": {"n_iter": 2},
    "iter2v": {"n_iter": 2, "vertical": True},  # the two bench winners combined
    # beta0 picked by the Thomson Te check (te_check.frame_err) alone, or with the edge miss
    "te4_e": {"n_iter": 4, "edge_err": True, "te_select": "te"},
    "teh4_e": {"n_iter": 4, "edge_err": True, "te_select": "hybrid"},
    "iter2_e": {"n_iter": 2, "edge_err": True},
    "iter1": {"n_iter": 1},
    "iter1_e": {"n_iter": 1, "edge_err": True},
    "iter0": {"n_iter": 0},  # J0 alone: the shape prior with no Picard pass
    "iter0_e": {"n_iter": 0, "edge_err": True},
    # post-hoc mast_vertical on a solved variant, as the c8v / iter4v MAST caches are built
    "pv": {"posthoc": "sym"},
    "iter4pv": {"posthoc": "iter4"},
    # frame-wise mean of two solved variants' psi, as an averaged MAST Final would be built
    "ens": {"ensemble": ("sym", "iter4pv")},
    # plasma flux (psi minus coil flux) smoothed over neighbouring frames, Gaussian sigma in frames
    "sym_ts1": {"tsmooth": ("sym", 1.0)},
    "iter4pv_ts1": {"tsmooth": ("iter4pv", 1.0)},
    "iter4pv_ts2": {"tsmooth": ("iter4pv", 2.0)},
    # fusion_eq.twin correction on a solved variant; model from FUSION_TWIN_MODEL
    "twin": {"twin": "iter4"},
    # correction scaled by c (picked on bench half A), optionally up-down symmetrised
    "twin_c": {"twin": "iter4", "scale": float(os.environ.get("FUSION_TWIN_C", 1.0))},
    "twin_cs": {"twin": "iter4", "scale": float(os.environ.get("FUSION_TWIN_C", 1.0)), "sym": True},
    "twin_cpv": {"posthoc": "twin_c"},
    "twin_c5": {"twin": "iter4", "scale": 0.5},
    "twin_c5pv": {"posthoc": "twin_c5"},
    # per-mode scaled twin (Twin.mode_scale); run each bench half with the model fitted on the other
    "twin_m": {"twin": "iter4"},
    "twin_mpv": {"posthoc": "twin_m"},
    # per-mode scale rules between twin_c (.39) and twin_m (s), cross-fit the same way: bLL = twin blend
    # --lam .LL --c .39; k85 = s where the mode's max|cos| with a MAST mode >= .85, else .39; f39 = max(s, .39)
    **{f"twin_{r}{s}": ({"posthoc": f"twin_{r}"} if s else {"twin": "iter4"})
       for r in ("b25", "b50", "b75", "k85", "f39") for s in ("", "pv")},
}


def load_row(path):
    import pyarrow.parquet as pq
    return {k: v[0] for k, v in pq.read_table(path).to_pydict().items()}


def edges(row, k):
    n = int(np.asarray(row["efit_lcfs_n"])[k])
    r = np.asarray(row["efit_lcfs_r"][k], float)[:n]
    return (r.min(), r.max()) if n >= 5 else (np.nan, np.nan)


def vacuum(row, frames):
    """(R, Z, t, vac, ip, s, mask): coil flux per frame and the Ip sign s the solver works in."""
    from data_fixes import fix_d3d_ip_times
    from .greens import coil_greens, interp_signal
    from .mast import machine_mask
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    t = np.asarray(row["efit_times"], float)[frames]
    tm = np.asarray(row["magnetics_time"], float)
    vac = sum(interp_signal(tm, row[c], t)[:, None, None] * 1e3 * g
              * (E_SCALE if c == "magnetics_ECOILA" else 1.0)
              for c, g in coil_greens(row, R, Z).items())
    ip = interp_signal(fix_d3d_ip_times(row), row["magnetics_plasma_current"], t) * 1e3
    s = np.sign(np.nanmedian(ip)) or 1.0  # solve with the axis a maximum, as on MAST
    return R, Z, t, vac, ip, s, machine_mask(row, R, Z)


def posthoc_vertical(row, frames, p):
    """mast_vertical's post-hoc force-balance shift applied to a solved bench psi (stored sign,
    any per-frame offset): the bench analogue of the MAST Finals built with mast_vertical.
    The running median spans 5 bench frames (every STRIDE-th EFIT frame), not 5 EFIT frames."""
    from scipy.ndimage import shift
    from .mast_vertical import vertical_shifts
    R, Z, _, vac, _, s, mc = vacuum(row, frames)
    out = -p  # solver sign (axis maximum); a per-frame constant leaves J and the shift unchanged
    z = vertical_shifts(out, s * vac, R, Z, mc)
    dZ = Z[1] - Z[0]
    return -np.stack([s * vac[k] + shift(out[k] - s * vac[k], (z[k] / dZ, 0), order=3, mode="nearest")
                      for k in range(len(p))])


def tsmooth(row, frames, p, sigma):
    """Gaussian smoothing of the plasma flux along the frame axis; each frame keeps its own coil
    flux. A per-frame offset in p is smoothed too, which moves no contour (the offset is refit)."""
    from scipy.ndimage import gaussian_filter1d
    _, _, _, vac, _, s, _ = vacuum(row, frames)
    plasma = p + s * vac  # p is stored sign (-solver), solver psi = s * vac + plasma part
    return gaussian_filter1d(plasma, sigma, axis=0, mode="nearest") - s * vac


def twin_correct(row, frames, p, scale=1.0, sym=False):
    """fusion_eq.twin's residual correction on a solved bench psi (stored sign, any offset),
    times `scale`; `sym` keeps only its up-down symmetric part."""
    import pickle
    from .twin import d3d_inputs, force_z
    tw = pickle.load(open(os.environ["FUSION_TWIN_MODEL"], "rb"))
    R, Z, G, amps, ip, rin, rout, mc = d3d_inputs(row, frames)
    vac = np.tensordot(amps, G, 1)
    psi = -p  # solver sign; the stored offset is a per-frame constant, removed by the grid mean
    ok = np.isfinite(rin) & (rout > rin) & (ip > 1e4)
    for k in np.flatnonzero(ok):
        phi = ((psi[k] - vac[k]) / ip[k]).reshape(1, -1)
        sc = np.array([[rin[k], rout[k], ip[k] / 1e6, force_z(psi[k], vac[k], R, Z, mc)]])
        d = tw.delta(phi, (vac[k] / ip[k]).reshape(1, -1), sc).reshape(psi[k].shape)
        psi[k] = psi[k] + scale * ip[k] * ((d + d[::-1]) / 2 if sym else d)
    return -psi


def predict(row, frames, vertical=False, alpha=1.0, kappa=1.8, delta=0.4, dbz=None, n_iter=8, edge_err=False,
            te_select=None):
    """psi (len(frames), nZ, nR) in DIII-D's stored sign, before the offset."""
    from .freeb import solve_matched
    from .mast import shaped_current
    R, Z, t, vac, ip, s, mc = vacuum(row, frames)
    out = s * vac
    for j, k in enumerate(frames):
        rin, rout = edges(row, k)
        if edge_err:  # same draw for every variant: seeded by shot and frame
            e = np.random.default_rng([int(row["efit_times"][0] * 1e6) % 2**31, int(k)])
            rin, rout = (rin + EDGE_ERR[0] + EDGE_ERR[1] * e.standard_normal(),
                         rout - EDGE_ERR[0] + EDGE_ERR[1] * e.standard_normal())
        if not (np.isfinite(rin) and np.isfinite(ip[j])):
            continue
        J = shaped_current(R, Z, (rin + rout) / 2, (rout - rin) / 2, kappa, delta, alpha=alpha)
        sel = None
        if te_select:
            from .te_check import frame_err, thomson_points
            pts = thomson_points(row, t[j])

            def sel(psi, edge, pts=pts):
                te = frame_err(psi, R, Z, pts)
                if not np.isfinite(te):
                    return edge
                # hybrid scales: Thomson edge sd ~0.02 m (EDGE_ERR); te spread between beta0s ~0.01
                return te + 1e-3 * edge if te_select == "te" else te / 0.01 + edge / 0.02
        out[j] = solve_matched(s * vac[j], J, R, Z, abs(ip[j]), mc, mc.astype(np.float64), rout,
                               alpha=alpha, vertical=vertical, n_iter=n_iter, select=sel,
                               **({"r_in": rin, "dbz": dbz} if dbz else {}))[0]
    return -out  # DIII-D stores the axis as a minimum


def _work(args):
    path, variant, out = args
    dst = Path(out) / (Path(path).stem + ".npz")
    if npz_done(dst):
        return
    row = load_row(path)
    T = len(row["efit_times"])
    frames = frames_of(T)
    psi_gt = np.stack([np.asarray(row["efit_psirz"][k], float) for k in frames])
    cfg = VARIANTS[variant]
    if "tsmooth" in cfg:
        base, sigma = cfg["tsmooth"]
        b = np.load(Path(out).parent / base / dst.name)
        assert (b["frames"] == frames).all()
        p = tsmooth(row, frames, b["psirz"].astype(np.float64), sigma)
    elif "ensemble" in cfg:
        bases = [np.load(Path(out).parent / v / dst.name) for v in cfg["ensemble"]]
        assert all((b["frames"] == frames).all() for b in bases)
        p = np.mean([b["psirz"].astype(np.float64) for b in bases], axis=0)
    elif "twin" in cfg:
        base = np.load(Path(out).parent / cfg["twin"] / dst.name)
        assert (base["frames"] == frames).all()
        p = twin_correct(row, frames, base["psirz"].astype(np.float64), cfg.get("scale", 1.0),
                         cfg.get("sym", False))
    elif "posthoc" in cfg:
        base = np.load(Path(out).parent / cfg["posthoc"] / dst.name)
        assert (base["frames"] == frames).all()
        p = posthoc_vertical(row, frames, base["psirz"].astype(np.float64))
    else:
        p = predict(row, frames, **cfg)
    p = p + (psi_gt - p).mean(axis=(1, 2), keepdims=True)
    save_npz(dst, psirz=p.astype(np.float32), frames=frames)


def _init():
    from .paths import use_starter
    use_starter()
    import local_score  # noqa: F401  puts fusion_scoring/ on sys.path


def _truth(path):
    row = load_row(path)
    frames = frames_of(len(row["efit_times"]))
    q95 = np.asarray(row["efit_q95"], float)[frames]
    bn = np.asarray(row["efit_beta_n"], float)[frames]
    return {"psi": np.stack([np.asarray(row["efit_psirz"][k], np.float32) for k in frames]),
            "q95": q95, "betaN": bn}


def _scored(args):
    """Per-shot scorer output for every variant (reference built once)."""
    path, variants, out = args
    import local_score as L
    mk = np.load(Path(L.HERE) / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z, mc = mk["grid_R"], mk["grid_Z"], mk["mask_coarse"].astype(bool)
    gt = _truth(path)
    ref = L.build_reference(gt["psi"], R, Z, mc, mc.astype(np.float64))
    from .greens import interp_signal
    from .qprofile import q_at, vacuum_f
    row = load_row(path)
    t = np.asarray(row["efit_times"], float)[frames_of(len(row["efit_times"]))]
    f = vacuum_f(interp_signal(row["magnetics_time"], row["magnetics_bcoil"], t) * 1e3, N_TF)
    preds = {}
    for v in variants:
        p = np.load(Path(out) / v / (Path(path).stem + ".npz"))["psirz"].astype(np.float64)
        q = np.array([q_at(p[k], R, Z, f[k], "DIII-D", mask_coarse=mc, mask_f=mc.astype(np.float64))
                      for k in range(len(p))])
        preds[v] = {"psirz": p, "q95": np.where(np.isfinite(q), q, np.nanmean(gt["q95"])),
                    "betaN": np.full_like(gt["betaN"], np.nanmean(gt["betaN"]))}
    return gt, ref, preds


def score(files, variants, out, workers, n_boot=1000, json_out=None):
    """Composite S per variant, and a paired shot-bootstrap CI of S(v) - S(variants[0]).
    `json_out` saves the full scores and every variant's S on the same bootstrap draws, so any
    pair can be compared later (fusion_eq.final_gate)."""
    _init()
    import local_score as L
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(_scored, [(f, variants, out) for f in files])
    R = np.load(Path(L.HERE) / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    Rg, Zg, mc = R["grid_R"], R["grid_Z"], R["mask_coarse"].astype(bool)
    ps, pss, pn = 0.0, 0.0, 0
    ss, sss, sn = np.zeros(L.N_SCALARS), np.zeros(L.N_SCALARS), np.zeros(L.N_SCALARS)
    cs, css, cn = np.zeros(L.N_CONS), np.zeros(L.N_CONS), np.zeros(L.N_CONS)
    for gt, ref, _ in res:
        g = gt["psi"].astype(np.float64)
        ps += g.sum(); pss += (g ** 2).sum(); pn += g.size
        for j, name in enumerate(L.SCALARS):
            v = gt[name][np.isfinite(gt[name])]
            ss[j] += v.sum(); sss[j] += (v ** 2).sum(); sn[j] += v.size
        for j in range(L.N_CONS):
            v = ref[1][ref[2][:, j], j]
            cs[j] += v.sum(); css[j] += (v ** 2).sum(); cn[j] += v.size
    stats = {"psi_sum": ps, "psi_sumsq": pss, "psi_n": pn, "scal_sum": ss, "scal_sumsq": sss,
             "scal_n": sn, "cons_sum": cs, "cons_sumsq": css, "cons_n": cn}
    means = (ps / pn, ss / np.maximum(sn, 1), cs / np.maximum(cn, 1))
    per = {v: [L.score_shot(gt, ref, p[v], Rg, Zg, mc, mc.astype(np.float64), 1, means)
               for gt, ref, p in res] for v in variants}

    def comp(shots, v):
        acc = L.Accum()
        acc.psi_sign = 1
        for i in shots:
            acc.add(per[v][i])
        return L.finalize_machine(acc, stats)  # stats from all shots: fine for paired deltas

    allx = np.arange(len(files))
    full = {v: comp(allx, v) for v in variants}
    for v in variants:
        r = full[v]
        print(f"{v:8s} S {r['S']:.4f}  psi {r['r2_psi']:.4f}  q95 {r['r2_each']['q95']:.4f}  dlcfs {r['dlcfs']:.4f}  "
              f"cons {r['consistency']:.4f}  " + " ".join(
                  f"{k} {x:.3f}" for k, x in r["r2_cons_each"].items() if x is not None))
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(files), len(files)) for _ in range(n_boot)]
    bs = {v: np.array([comp(b, v)["S"] for b in boots]) for v in variants}
    for v in variants[1:]:
        d = bs[v] - bs[variants[0]]
        print(f"{v} - {variants[0]}: dS {full[v]['S'] - full[variants[0]]['S']:+.4f}  "
              f"95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
    if json_out:
        import json
        Path(json_out).write_text(json.dumps({
            "n_shots": len(files), "n_boot": n_boot, "files": [Path(f).stem for f in files],
            "S": {v: full[v]["S"] for v in variants},
            "cons_each": {v: full[v]["r2_cons_each"] for v in variants},
            "boot_S": {v: bs[v].tolist() for v in variants}}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="text file, one parquet path per line")
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--score", nargs="+", help="variant dirs under --out, baseline first")
    ap.add_argument("--json", help="with --score: save scores and bootstrap draws here")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    files = [l.strip() for l in open(a.files) if l.strip()]
    if a.score:
        score(files, a.score, a.out, a.workers, n_boot=a.n_boot, json_out=a.json)
        return 0
    out = Path(a.out) / a.variant
    out.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers, initializer=_init) as pool:
        for _ in pool.imap_unordered(_work, [(f, a.variant, out) for f in files]):
            pass
    print(a.variant, len(list(out.glob("*.npz"))), "shots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
