"""Zero-shot MAST predictor from physics: coil Green's functions plus a modelled plasma current.

No MAST label enters the model. Plasma position comes from Thomson scattering, which on MAST is a
horizontal midplane laser: where the electron density collapses marks the boundary's inner and
outer midplane radii. Shape priors (elongation, triangularity, profile peaking) are arguments so
they can be studied; the defaults are generic spherical-tokamak values, not fitted.
"""
import numpy as np

from .greens import MU0, interp_signal, plasma_psi, vacuum_psi
from .freeb import solve, solve_matched
from .qprofile import q_at, vacuum_f

# The solenoid column reads exactly 2x the FAIR-MAST level-2 pf_active solenoid current, an
# open convention question on the Data page; its 656 conductor rows are single-turn elements.
# 0.5 recovers the level-2 current and is the default. With the convention open, pick_sol_scale
# chooses 0.5, 0.75 or 1.0 per shot from the Thomson inner edge (MAST inputs only).
SOL_SCALE = 0.5
# Candidates for the per-shot fit to the Thomson inner edge (pick_sol_scale).
SOL_SCALES = (0.5, 0.75, 1.0)
N_TF = 24  # MAST TF centre-rod limbs, one turn each (assumed from MAST design papers, not shipped)
# Share of MAST stored energy carried by NBI fast ions, which Thomson cannot see: ~35% per
# Cecconello et al., Nucl. Fusion 59 016006 (2019), citing TRANSP/NUBEAM. Literature, not fitted.
FAST_ION_FRAC = 0.35
E_CHARGE = 1.602176634e-19
# Inboard limiter (centre-column armour) radius, taken as the inner edge of MAST's design plasma,
# R - a = 0.85 - 0.65 = 0.20 m (Sykes et al., IAEA FEC 2000, OV4/1, Table 1); the armour is 18 mm
# of graphite outside the Inconel centre tube (Edlington et al., Rev. Sci. Instrum. 72, 421
# (2001)). Machine geometry, not fitted.
WALL_R = 0.20
# Frames per running mean of the plasma flux and q95. Successive EFIT frames are 5 ms apart, much
# shorter than the current-diffusion time, so frame-to-frame scatter in the solve is noise.
# The scatter enters through the Thomson outer edge that places each solve; leave-one-out CV of a
# running mean over that edge on all 1206 public test shots (mast_smooth_cv, MAST inputs only)
# picks 7 Thomson pulses = 29 ms = 5 EFIT frames.
SMOOTH = 5
# Frames per running mean of betaN. Thomson pressure scatters far more frame to frame than the
# stored energy can change (confinement time tens of ms); 9 frames = 45 ms. Demo betaN R2
# 0.39 -> 0.55, rising on all 3 shots (wider windows rise further but approach a shot mean).
BETA_SMOOTH = 9


def _median5(x):
    """5-channel running median along the last axis; rejects single-channel Thomson spikes."""
    pad = np.pad(x, ((0, 0), (2, 2)), constant_values=np.nan)
    return np.nanmedian(np.stack([pad[:, k:k + x.shape[1]] for k in range(5)]), axis=0)


def thomson_edges(row, times, frac: float = 0.3, raw: bool = False):
    """(R_in, R_out) per time (ms): outermost channels whose 5-channel median density exceeds
    `frac` of the profile's 90th percentile. NaN where the profile is empty. `raw` returns them
    per Thomson pulse instead, without interpolating over empty pulses."""
    from data_fixes import align_mast_thomson_core
    R, _, ne = align_mast_thomson_core(row)
    tt = np.asarray(row["thomson_core_times"], float)
    order = np.argsort(R)
    R, ne = R[order], ne[:, order]
    med = _median5(ne)
    rin, rout = np.full(len(tt), np.nan), np.full(len(tt), np.nan)
    for i, m in enumerate(med):
        if not np.isfinite(m).any():
            continue
        ok = np.where(m > frac * np.nanpercentile(m, 90))[0]
        if ok.size:
            rin[i], rout[i] = R[ok[0]], R[ok[-1]]
    good = np.isfinite(rin)
    if raw:
        return rin, rout
    if not good.any():
        return np.full(len(times), np.nan), np.full(len(times), np.nan)
    return np.interp(times, tt[good], rin[good]), np.interp(times, tt[good], rout[good])


def thomson_te(row, times):
    """(R_ch ascending, Te (len(times), n_ch) eV) from the core Thomson frame nearest each time."""
    from data_fixes import align_mast_thomson_core
    R, Te, _ = align_mast_thomson_core(row)
    tt = np.asarray(row["thomson_core_times"], float)
    order = np.argsort(R)
    k = np.abs(tt[None] - np.asarray(times, float)[:, None]).argmin(1)
    return R[order], np.asarray(Te, float)[k][:, order]


def spitzer_profile(R_ch, te, R, Z, n_bins: int = 10, min_ch: int = 5):
    """Profile callable for freeb.solve: J shape Te^1.5 (Spitzer conductivity, E constant on a
    relaxed ohmic plasma), with Te put on psi_N through the midplane Thomson channels of the
    current solve. Binned medians, forced non-increasing outward, 0 at the LCFS. Falls back to
    1 - psi_N when fewer than `min_ch` channels with Te lie inside the LCFS."""
    iz = int(np.argmin(np.abs(Z)))  # Thomson chords sit at Z = 0
    edges = np.linspace(0, 1, n_bins + 1)

    def f(pn, inside):
        x = np.interp(R_ch, R, pn[iz])
        ok = (np.interp(R_ch, R, inside[iz].astype(float)) > 0.5) & np.isfinite(te) & (te > 0)
        if ok.sum() < min_ch:
            return 1 - pn
        b = np.clip(np.digitize(x[ok], edges) - 1, 0, n_bins - 1)
        xs = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins) if (b == i).any()]
        ys = np.minimum.accumulate([np.median(te[ok][b == i]) for i in range(n_bins) if (b == i).any()])
        ys = (ys / ys[0]) ** 1.5
        return np.interp(pn, np.append(xs, 1.0), np.append(ys, 0.0))
    return f


def machine_mask(row, R, Z, clearance: float = 0.05) -> np.ndarray:
    """(nZ, nR) bool: grid nodes inside the convex hull of the shipped coil conductors, outboard
    of the solenoid and more than `clearance` from any conductor. Built from machine geometry
    only; it replaces the scorer's label-derived envelope wherever the solver searches for the
    axis and LCFS (Rule 3 keeps that envelope inside the scoring derivation)."""
    from scipy.spatial import ConvexHull, Delaunay
    cR, cZ = np.asarray(row["coil_R"], float), np.asarray(row["coil_Z"], float)
    sol = np.asarray(row["coil_input_column"]) == "magnetics_sol_current"
    Rg, Zg = np.meshgrid(R, Z)
    q = np.c_[Rg.ravel(), Zg.ravel()]
    pts = np.c_[cR, cZ]
    inside = Delaunay(pts[ConvexHull(pts).vertices]).find_simplex(q) >= 0
    gap = np.min(np.hypot(q[:, :1] - cR[None], q[:, 1:] - cZ[None]), axis=1)
    return (inside & (gap > clearance) & (q[:, 0] > cR[sol].max())).reshape(Rg.shape)


def shaped_current(R, Z, r0, a, kappa, delta, z0=0.0, alpha=1.0):
    """J (nZ, nR) ∝ (1 - rho^2)^alpha inside a D-shaped boundary.

    rho is the node's distance from (r0, z0) over the boundary's distance along the same ray, in
    coordinates where Z is divided by kappa. The boundary is Miller's R = r0 + a cos(t + asin(delta)
    sin t), Z = z0 + kappa a sin t, evaluated at t = the node's angle (exact for delta = 0).
    """
    Rg, Zg = np.meshgrid(R, Z)
    dz = (Zg - z0) / kappa
    t = np.arctan2(dz, Rg - r0)
    bR = a * np.cos(t + np.arcsin(np.clip(delta, -0.9, 0.9)) * np.sin(t))
    bZ = a * np.sin(t)
    rho = np.hypot(Rg - r0, dz) / np.maximum(np.hypot(bR, bZ), 1e-9)
    return np.where(rho < 1, (1 - np.minimum(rho, 1) ** 2) ** alpha, 0.0)


def q95_iter(r0, a, b0, ip_ma, kappa, delta):
    """ITER Physics Basis (1999) eq. for q95 with the aspect-ratio correction; valid down to
    eps ~ 0.7, which covers MAST's eps ~ 0.65."""
    eps = a / r0
    shape = (1 + kappa ** 2 * (1 + 2 * delta ** 2 - 1.2 * delta ** 3)) / 2
    return 5 * a ** 2 * b0 / (r0 * ip_ma) * shape * (1.17 - 0.65 * eps) / (1 - eps ** 2) ** 2


def mean_pressure(row, times, rin, rout, ti_over_te: float = 1.0, raw: bool = False):
    """Volume-averaged thermal pressure (Pa) at `times` from midplane Thomson; `raw` returns it
    per Thomson pulse, NaN where a pulse has no usable profile.

    Channels map to rho = |R - r0| / a; the profile is averaged over the cross-section with
    weight 2 rho d rho (exact for nested similar ellipses). p = ne e Te (1 + Ti/Te), ni = ne."""
    from data_fixes import align_mast_thomson_core
    R, Te, ne = align_mast_thomson_core(row)
    tt = np.asarray(row["thomson_core_times"], float)
    order = np.argsort(R)
    R, p = R[order], _median5((ne * Te)[:, order]) * E_CHARGE * (1 + ti_over_te)
    r_in, r_out = np.interp(tt, times, rin), np.interp(tt, times, rout)
    grid = np.linspace(0, 1, 51)
    out = np.full(len(tt), np.nan)
    for i in range(len(tt)):
        if not (np.isfinite(r_in[i]) and r_out[i] > r_in[i]):
            continue
        r0, a = (r_in[i] + r_out[i]) / 2, (r_out[i] - r_in[i]) / 2
        rho, pi = np.abs(R - r0) / a, p[i]
        ok = np.isfinite(pi) & (rho <= 1)
        if ok.sum() < 5:
            continue
        o = np.argsort(rho[ok])
        prof = np.interp(grid, rho[ok][o], pi[ok][o])
        out[i] = np.trapezoid(prof * 2 * grid, grid)
    good = np.isfinite(out)
    if raw:
        return out
    return np.interp(times, tt[good], out[good]) if good.any() else np.full(len(times), np.nan)


def mast_scalars(row, t, rin, rout, kappa, delta, ti_over_te: float = 1.0):
    """(q95, betaN) per time from Ip, the TF current and the Thomson boundary and pressure."""
    ip = np.abs(interp_signal(row["magnetics_time"], row["magnetics_plasma_current"], t)) * 1e-3
    itf = np.abs(interp_signal(row["magnetics_time"], row["magnetics_tf_current"], t)) * 1e3
    r0, a = (rin + rout) / 2, (rout - rin) / 2
    b0 = MU0 * N_TF * itf / (2 * np.pi * r0)
    q95 = q95_iter(r0, a, b0, np.maximum(ip, 1e-3), kappa, delta)
    beta_t = mean_pressure(row, t, rin, rout, ti_over_te) / (b0 ** 2 / (2 * MU0))
    beta_n = 100 * beta_t * a * b0 / np.maximum(ip, 1e-3)
    # Frames without a Thomson boundary get the shot median rather than NaN (scored as residual).
    for x, d in ((q95, 6.0), (beta_n, 1.0)):
        bad = ~np.isfinite(x)
        x[bad] = np.nanmedian(x) if (~bad).any() else d
    return q95, beta_n


def total_beta_n(thermal, fast_ion_frac: float = FAST_ION_FRAC):
    """Thermal betaN -> total (EFIT-like) betaN, then a running median: the Thomson pressure
    jumps frame to frame far more than the equilibrium does."""
    x = robust_series(np.asarray(thermal, float) / (1 - fast_ion_frac), 0.01, 10)
    return np.where(np.isfinite(x), x, thermal)


def robust_series(x, lo, hi, win: int = 5):
    """Clip to a physical range, then a running median in time: one broken LCFS frame should
    not become a q95 spike. NaNs are filled from neighbours, or `fill` if the shot has none."""
    x = np.where((x >= lo) & (x <= hi), x, np.nan)
    if not np.isfinite(x).any():
        return np.full_like(x, np.nan)
    h = win // 2
    pad = np.pad(x, h, mode="edge")
    med = np.nanmedian(np.stack([pad[k:k + len(x)] for k in range(win)]), axis=0)
    ok = np.isfinite(med)
    return np.interp(np.arange(len(x)), np.flatnonzero(ok), med[ok])


def smooth_frames(x, win: int = SMOOTH):
    """Centred running mean along axis 0, edge frames repeated."""
    if win <= 1:
        return x
    h = win // 2
    pad = np.concatenate([x[:1].repeat(h, 0), x, x[-1:].repeat(h, 0)])
    return np.mean([pad[k:k + len(x)] for k in range(win)], axis=0)


def finalize(row, psi, vac, q95_shape, win: int = SMOOTH) -> tuple[np.ndarray, np.ndarray]:
    """Time-smooth the plasma part of psi, then measure q95 on it (falling back to the shape
    estimate where no surface is found). Shared by predict_mast and mast_post."""
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    mc = machine_mask(row, R, Z)
    t = np.asarray(row["efit_times"], float)
    psi = vac + smooth_frames(np.asarray(psi, float) - vac, win)
    f = vacuum_f(interp_signal(row["magnetics_time"], row["magnetics_tf_current"], t) * 1e3, N_TF)
    qn = robust_series(np.array([q_at(psi[k], R, Z, f[k], "MAST", mask_coarse=mc,
                                      mask_f=mc.astype(np.float64), wall_r=WALL_R)
                                 for k in range(len(t))]), 2, 25)
    q95 = np.where(np.isfinite(qn), qn, q95_shape)
    return psi, smooth_frames(q95, win)


def pick_sol_scale(row, scales=SOL_SCALES, n_frames: int = 3, kappa: float = 1.8,
                   delta: float = 0.4) -> float:
    """Per-shot solenoid scale whose solved LCFS inner midplane radius best matches the Thomson
    inner density edge, on `n_frames` frames spread over the shot.

    The organizers leave the 2x solenoid factor open (Data page), and on shots with P2 off the
    0.5 scale puts the LCFS on the centre column 9-20 cm inside the Thomson edge, while the
    full scale lands within 3 cm. The outer edge is matched separately through beta0
    (solve_matched), so the inner edge is the remaining free constraint. Uses MAST inputs only."""
    from lcfs import extract_lcfs
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    t = np.asarray(row["efit_times"], float)
    t = t[np.linspace(len(t) // 4, 3 * len(t) // 4, n_frames).astype(int)]
    rin, rout = thomson_edges(row, t)
    ok = np.isfinite(rin) & (rout > rin)
    if not ok.any():
        return SOL_SCALE
    t, rin, rout = t[ok], rin[ok], rout[ok]
    mc = machine_mask(row, R, Z)
    ip = interp_signal(row["magnetics_time"], row["magnetics_plasma_current"], t) * 1e3
    err = []
    for s in scales:
        vac = vacuum_psi(row, R, Z, t, {"magnetics_sol_current": s})
        e = []
        for k in range(len(t)):
            J = shaped_current(R, Z, (rin[k] + rout[k]) / 2, (rout[k] - rin[k]) / 2, kappa, delta)
            psi = solve_matched(vac[k], J, R, Z, ip[k], mc, mc.astype(np.float64), rout[k])[0]
            c = extract_lcfs(psi, R, Z, "MAST", mc, mc.astype(np.float64), n_iter=14, n_points=256)
            e.append(abs(c[:, 0].min() - rin[k]) if c is not None else np.inf)
        err.append(np.mean(e))
    return float(scales[int(np.argmin(err))])


def predict_mast(row, sol_scale: float | str = SOL_SCALE, kappa: float = 1.8, delta: float = 0.4,
                 alpha: float = 1.0, frac: float = 0.3, ti_over_te: float = 1.0,
                 q_from_psi: bool = True, freeb_iter: int = 8, beta0: float = -1.0,
                 fast_ion_frac: float = FAST_ION_FRAC, profile: str = "power",
                 vertical: bool = False, warm: bool = False) -> dict:
    """`profile` "spitzer" takes the J shape from Thomson Te (spitzer_profile); `vertical` holds
    each frame at coil force balance inside the solve (freeb.solve); `warm` searches beta0 only
    within 0.1 of the previous frame's value. J itself always restarts from the Thomson-centred
    shape: carrying J over lets the Spitzer mapping walk the axis outward frame after frame
    (shot 0 axis 0.84 -> 1.70 m), while a fresh start anchors every frame to the measured edges."""
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    mc = machine_mask(row, R, Z)
    t = np.asarray(row["efit_times"], float)
    if sol_scale == "thomson":
        sol_scale = pick_sol_scale(row, kappa=kappa, delta=delta)
    vac = vacuum_psi(row, R, Z, t, {"magnetics_sol_current": sol_scale})
    ip = interp_signal(row["magnetics_time"], row["magnetics_plasma_current"], t) * 1e3
    rin, rout = thomson_edges(row, t, frac)
    if profile == "spitzer":
        R_ch, te = thomson_te(row, t)
    psi = vac.copy()
    prev = None  # beta0 of the last solved frame, for the warm start
    beta0s = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    for k in range(len(t)):
        if not (np.isfinite(rin[k]) and rout[k] > rin[k]):
            prev = None
            continue
        J = shaped_current(R, Z, (rin[k] + rout[k]) / 2, (rout[k] - rin[k]) / 2, kappa, delta,
                           alpha=alpha)
        kw = {"vertical": vertical}
        if profile == "spitzer":
            kw["profile"] = spitzer_profile(R_ch, te[k], R, Z)
        bs = beta0s
        if warm and prev is not None:
            bs = tuple(b for b in beta0s if abs(b - prev) < 0.15)
        if freeb_iter and beta0 < 0:  # negative = match Thomson outer edge per frame
            psi[k], _, prev = solve_matched(vac[k], J, R, Z, ip[k], mc, mc.astype(np.float64),
                                            rout[k], alpha=alpha, beta0s=bs, n_iter=freeb_iter, **kw)
        elif freeb_iter:
            psi[k] = solve(vac[k], J, R, Z, ip[k], mc, mc.astype(np.float64), alpha=alpha,
                           beta0=beta0, n_iter=freeb_iter, **kw)[0]
        else:
            psi[k] += plasma_psi(J, R, Z, ip[k])
    q95, beta_n = mast_scalars(row, t, rin, rout, kappa, delta, ti_over_te)
    beta_n = smooth_frames(total_beta_n(beta_n, fast_ion_frac), BETA_SMOOTH)
    if q_from_psi:
        psi, q95 = finalize(row, psi, vac, q95)
    return {"psirz": psi.astype(np.float32), "q95": q95.astype(np.float32),
            "betaN": beta_n.astype(np.float32), "sol_scale": float(sol_scale)}
