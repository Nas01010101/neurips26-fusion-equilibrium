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
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .paths import npz_done, save_npz

E_SCALE = 65.0
STRIDE = 10  # every 10th EFIT frame
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
}


def load_row(path):
    import pyarrow.parquet as pq
    return {k: v[0] for k, v in pq.read_table(path).to_pydict().items()}


def edges(row, k):
    n = int(np.asarray(row["efit_lcfs_n"])[k])
    r = np.asarray(row["efit_lcfs_r"][k], float)[:n]
    return (r.min(), r.max()) if n >= 5 else (np.nan, np.nan)


def predict(row, frames, vertical=False, alpha=1.0, kappa=1.8, delta=0.4, dbz=None, n_iter=8, edge_err=False,
            te_select=None):
    """psi (len(frames), nZ, nR) in DIII-D's stored sign, before the offset."""
    from data_fixes import fix_d3d_ip_times
    from .freeb import solve_matched
    from .greens import coil_greens, interp_signal
    from .mast import machine_mask, shaped_current
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    t = np.asarray(row["efit_times"], float)[frames]
    tm = np.asarray(row["magnetics_time"], float)
    vac = sum(interp_signal(tm, row[c], t)[:, None, None] * 1e3 * g
              * (E_SCALE if c == "magnetics_ECOILA" else 1.0)
              for c, g in coil_greens(row, R, Z).items())
    ip = interp_signal(fix_d3d_ip_times(row), row["magnetics_plasma_current"], t) * 1e3
    s = np.sign(np.nanmedian(ip)) or 1.0  # solve with the axis a maximum, as on MAST
    mc = machine_mask(row, R, Z)
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
    frames = np.arange(0, T, STRIDE)
    psi_gt = np.stack([np.asarray(row["efit_psirz"][k], float) for k in frames])
    p = predict(row, frames, **VARIANTS[variant])
    p = p + (psi_gt - p).mean(axis=(1, 2), keepdims=True)
    save_npz(dst, psirz=p.astype(np.float32), frames=frames)


def _init():
    from .paths import use_starter
    use_starter()
    import local_score  # noqa: F401  puts fusion_scoring/ on sys.path


def _truth(path):
    row = load_row(path)
    frames = np.arange(0, len(row["efit_times"]), STRIDE)
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
    t = np.asarray(row["efit_times"], float)[np.arange(0, len(row["efit_times"]), STRIDE)]
    f = vacuum_f(interp_signal(row["magnetics_time"], row["magnetics_bcoil"], t) * 1e3, N_TF)
    preds = {}
    for v in variants:
        p = np.load(Path(out) / v / (Path(path).stem + ".npz"))["psirz"].astype(np.float64)
        q = np.array([q_at(p[k], R, Z, f[k], "DIII-D", mask_coarse=mc, mask_f=mc.astype(np.float64))
                      for k in range(len(p))])
        preds[v] = {"psirz": p, "q95": np.where(np.isfinite(q), q, np.nanmean(gt["q95"])),
                    "betaN": np.full_like(gt["betaN"], np.nanmean(gt["betaN"]))}
    return gt, ref, preds


def score(files, variants, out, workers, n_boot=1000):
    """Composite S per variant, and a paired shot-bootstrap CI of S(v) - S(variants[0])."""
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
    for v in variants[1:]:
        d = np.array([comp(b, v)["S"] - comp(b, variants[0])["S"] for b in boots])
        print(f"{v} - {variants[0]}: dS {full[v]['S'] - full[variants[0]]['S']:+.4f}  "
              f"95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="text file, one parquet path per line")
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--score", nargs="+", choices=sorted(VARIANTS), help="baseline first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    files = [l.strip() for l in open(a.files) if l.strip()]
    if a.score:
        score(files, a.score, a.out, a.workers)
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
