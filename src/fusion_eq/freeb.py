"""Fixed-coil free-boundary equilibrium by Picard iteration on the plasma current.

The plasma boundary is not prescribed: each pass puts J on the flux surfaces of the current total
psi (coils + plasma), so the coil currents and Ip set the shape. Profile shape follows FreeGS's
"constrain Ip" family, J = (beta0 R/R0 + (1 - beta0) R0/R) (1 - psi_N)^alpha, normalised to Ip.
No equilibrium label enters; beta0 and alpha are generic priors.

Up-down symmetrising J is what keeps the iteration stable: an elongated plasma is vertically
unstable and a free Picard loop drifts into the divertor. MAST runs mostly double-null, so the
symmetric solution is also the physical one for most of the database.
"""
import numpy as np

from .greens import plasma_psi


def _psi_n(psi, R, Z, mask_coarse, mask_f):
    """(psi_N grid, inside-LCFS bool grid) for the scorer's LCFS, or (None, None) on failure."""
    from derive import _bilinear, _points_in_polygon, magnetic_axis
    from lcfs import extract_lcfs
    lcfs = extract_lcfs(psi, R, Z, "MAST", mask_coarse, mask_f, n_iter=14, n_points=256)
    if lcfs is None:
        return None, None
    ra, za, _, _ = magnetic_axis(psi, R, Z, "MAST", mask_coarse)
    if not np.isfinite(ra):
        return None, None
    p_ax = float(_bilinear(psi, R, Z, np.array([ra]), np.array([za]))[0])
    p_b = float(np.mean(_bilinear(psi, R, Z, lcfs[:, 0], lcfs[:, 1])))
    if p_ax == p_b:
        return None, None
    Rg, Zg = np.meshgrid(R, Z)
    inside = _points_in_polygon(Rg.ravel(), Zg.ravel(), lcfs).reshape(Rg.shape)
    return np.clip((psi - p_ax) / (p_b - p_ax), 0, 1), inside


def solve(vac, J0, R, Z, ip, mask_coarse, mask_f, alpha=1.0, beta0=0.5, n_iter=12,
          relax=0.5, symmetric=True):
    """Return (psi, J) after `n_iter` Picard passes from the initial current J0 (nZ, nR).

    A pass that cannot find a closed LCFS keeps the previous J, so the result is never worse
    defined than the initial guess."""
    Rg = np.meshgrid(R, Z)[0]
    J = J0 / max(J0.sum(), 1e-30)
    psi = vac + plasma_psi(J, R, Z, ip)
    for _ in range(n_iter):
        pn, inside = _psi_n(psi, R, Z, mask_coarse, mask_f)
        if pn is None:
            break
        r0 = float((Rg * inside).sum() / max(inside.sum(), 1))
        Jn = np.where(inside, (beta0 * Rg / r0 + (1 - beta0) * r0 / Rg) * (1 - pn) ** alpha, 0.0)
        if symmetric:
            Jn = (Jn + Jn[::-1]) / 2
        if Jn.sum() <= 0:
            break
        J = (1 - relax) * J + relax * Jn / Jn.sum()
        psi = vac + plasma_psi(J, R, Z, ip)
    return psi, J


def solve_matched(vac, J0, R, Z, ip, mask_coarse, mask_f, r_out, alpha=1.0,
                  beta0s=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5), n_iter=8):
    """Solve for each beta0 and keep the one whose LCFS outer midplane radius is nearest the
    Thomson-measured edge `r_out`. beta0 sets the pressure share of J and so the outward radial
    force; a measured boundary pins it, as magnetics + kinetic constraints do inside EFIT."""
    from lcfs import extract_lcfs
    best, best_err = None, np.inf
    for b in beta0s:
        psi, J = solve(vac, J0, R, Z, ip, mask_coarse, mask_f, alpha=alpha, beta0=b, n_iter=n_iter)
        c = extract_lcfs(psi, R, Z, "MAST", mask_coarse, mask_f, n_iter=14, n_points=256)
        err = abs(c[:, 0].max() - r_out) if c is not None else np.inf
        if best is None or err < best_err:
            best, best_err = (psi, J), err
    return best
