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
          relax=0.5, symmetric=True, profile=None, vertical=False):
    """Return (psi, J) after `n_iter` Picard passes from the initial current J0 (nZ, nR).

    `profile(pn, inside)` replaces the (1 - psi_N)^alpha shape when given. `vertical` replaces
    the up-down symmetrisation by a rigid shift of each new J to where the coils' vertical force
    on it vanishes (mast_vertical.force_zero): that is where MAST's radial-field feedback holds
    the plasma, and it lets the solved shape go up-down asymmetric. A pass with no force zero
    within reach falls back to symmetrising.

    A pass that cannot find a closed LCFS keeps the previous J, so the result is never worse
    defined than the initial guess."""
    Rg, Zg = np.meshgrid(R, Z)
    if vertical:  # lazy: mast_vertical imports mast, which imports this module
        from scipy.ndimage import shift
        from .mast_vertical import Z_MAX, force_zero
        dZ = Z[1] - Z[0]
        dvac = np.gradient(vac, dZ, axis=0)
    J = J0 / max(J0.sum(), 1e-30)
    psi = vac + plasma_psi(J, R, Z, ip)
    for _ in range(n_iter):
        pn, inside = _psi_n(psi, R, Z, mask_coarse, mask_f)
        if pn is None:
            break
        r0 = float((Rg * inside).sum() / max(inside.sum(), 1))
        shape = profile(pn, inside) if profile else (1 - pn) ** alpha
        Jn = np.where(inside, (beta0 * Rg / r0 + (1 - beta0) * r0 / Rg) * shape, 0.0)
        z = force_zero(Jn, dvac, dZ) if vertical and Jn.sum() > 0 else np.nan
        if np.isfinite(z):  # clip the resulting centroid, not the step
            zc = float((Zg * Jn).sum() / Jn.sum())
            Jn = np.maximum(shift(Jn, ((np.clip(zc + z, -Z_MAX, Z_MAX) - zc) / dZ, 0), order=1,
                                  mode="constant"), 0)
        elif symmetric:
            Jn = (Jn + Jn[::-1]) / 2
        if Jn.sum() <= 0:
            break
        J = (1 - relax) * J + relax * Jn / Jn.sum()
        psi = vac + plasma_psi(J, R, Z, ip)
    return psi, J


def solve_matched(vac, J0, R, Z, ip, mask_coarse, mask_f, r_out, alpha=1.0,
                  beta0s=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5), n_iter=8, r_in=None, dbz=(0.0,), select=None,
                  **kw):
    """Solve for each beta0 and keep the one whose LCFS outer midplane radius is nearest the
    Thomson-measured edge `r_out`. beta0 sets the pressure share of J and so the outward radial
    force; a measured boundary pins it, as magnetics + kinetic constraints do inside EFIT.

    With `r_in` the inner edge is matched too, through a uniform vertical field dBz (T) added to
    the vacuum (psi += dBz R^2 / 2 in Wb/rad): an error field of the kind vessel currents leave,
    which moves the plasma radially while beta0 sets its outward push. The error is then the sum
    of both edge misses, and the added field stays in the returned psi.
    `select(psi, edge_err)` replaces the selection error, e.g. to weigh in a Thomson Te check.
    Returns (psi, J, beta0); `kw` goes to solve."""
    from lcfs import extract_lcfs
    R2 = np.broadcast_to(R[None, :] ** 2 / 2, vac.shape)
    best, best_err = None, np.inf
    for bz in dbz if r_in is not None else (0.0,):
        v = vac + bz * R2 if bz else vac
        for b in beta0s:
            psi, J = solve(v, J0, R, Z, ip, mask_coarse, mask_f, alpha=alpha, beta0=b,
                           n_iter=n_iter, **kw)
            c = extract_lcfs(psi, R, Z, "MAST", mask_coarse, mask_f, n_iter=14, n_points=256)
            if c is None:
                err = np.inf
            else:
                err = abs(c[:, 0].max() - r_out)
                if r_in is not None:
                    err += abs(c[:, 0].min() - r_in)
            if select is not None:
                err = select(psi, err)
            if best is None or err < best_err:
                best, best_err = (psi, J, b), err
    return best
