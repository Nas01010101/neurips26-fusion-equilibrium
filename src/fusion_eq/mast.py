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

N_TF = 24  # MAST TF centre-rod limbs, one turn each (assumed from MAST design papers, not shipped)
# Share of MAST stored energy carried by NBI fast ions, which Thomson cannot see: ~35% per
# Cecconello et al., Nucl. Fusion 59 016006 (2019), citing TRANSP/NUBEAM. Literature, not fitted.
FAST_ION_FRAC = 0.35
E_CHARGE = 1.602176634e-19


def _median5(x):
    """5-channel running median along the last axis; rejects single-channel Thomson spikes."""
    pad = np.pad(x, ((0, 0), (2, 2)), constant_values=np.nan)
    return np.nanmedian(np.stack([pad[:, k:k + x.shape[1]] for k in range(5)]), axis=0)


def thomson_edges(row, times, frac: float = 0.3):
    """(R_in, R_out) per time (ms): outermost channels whose 5-channel median density exceeds
    `frac` of the profile's 90th percentile. NaN where the profile is empty."""
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
    if not good.any():
        return np.full(len(times), np.nan), np.full(len(times), np.nan)
    return np.interp(times, tt[good], rin[good]), np.interp(times, tt[good], rout[good])


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


def mean_pressure(row, times, rin, rout, ti_over_te: float = 1.0):
    """Volume-averaged thermal pressure (Pa) at `times` from midplane Thomson.

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


def predict_mast(row, sol_scale: float = 1.0, kappa: float = 1.8, delta: float = 0.4,
                 alpha: float = 1.0, frac: float = 0.3, ti_over_te: float = 1.0,
                 q_from_psi: bool = True, freeb_iter: int = 8, beta0: float = -1.0,
                 fast_ion_frac: float = FAST_ION_FRAC) -> dict:
    from .paths import STARTER
    mask = np.load(STARTER / "fusion_scoring" / "masks" / "mast_envelope.npz")
    mc = mask["mask_coarse"].astype(bool)
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    t = np.asarray(row["efit_times"], float)
    vac = vacuum_psi(row, R, Z, t, {"magnetics_sol_current": sol_scale})
    ip = interp_signal(row["magnetics_time"], row["magnetics_plasma_current"], t) * 1e3
    rin, rout = thomson_edges(row, t, frac)
    psi = vac.copy()
    for k in range(len(t)):
        if not (np.isfinite(rin[k]) and rout[k] > rin[k]):
            continue
        J = shaped_current(R, Z, (rin[k] + rout[k]) / 2, (rout[k] - rin[k]) / 2, kappa, delta,
                           alpha=alpha)
        if freeb_iter and beta0 < 0:  # negative = match Thomson outer edge per frame
            psi[k] = solve_matched(vac[k], J, R, Z, ip[k], mc, mc.astype(np.float64), rout[k],
                                   alpha=alpha, n_iter=freeb_iter)[0]
        elif freeb_iter:
            psi[k] = solve(vac[k], J, R, Z, ip[k], mc, mc.astype(np.float64), alpha=alpha,
                           beta0=beta0, n_iter=freeb_iter)[0]
        else:
            psi[k] += plasma_psi(J, R, Z, ip[k])
    q95, beta_n = mast_scalars(row, t, rin, rout, kappa, delta, ti_over_te)
    beta_n = total_beta_n(beta_n, fast_ion_frac)
    if q_from_psi:
        # Measured on the 3 demo shots' true psi: matches EFIT q95 within 0.1 past the first frame.
        f = vacuum_f(interp_signal(row["magnetics_time"], row["magnetics_tf_current"], t) * 1e3, N_TF)
        qn = robust_series(np.array([q_at(psi[k], R, Z, f[k], "MAST", mask_coarse=mc,
                                          mask_f=mc.astype(np.float64)) for k in range(len(t))]), 2, 25)
        q95 = np.where(np.isfinite(qn), qn, q95)
    return {"psirz": psi.astype(np.float32), "q95": q95.astype(np.float32),
            "betaN": beta_n.astype(np.float32)}
