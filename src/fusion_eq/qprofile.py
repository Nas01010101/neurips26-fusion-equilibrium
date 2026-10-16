"""Safety factor from a flux map: q(psi_N) = F / (2 pi) * closed-integral dl / (R |grad psi|).

psi is EFIT's Wb/rad, so B_p = |grad psi| / R and B_t = F / R. Uses the competition scorer's own
axis and LCFS finders so q95 is measured on the same surfaces the scorer would extract.
"""
import numpy as np

from .greens import MU0


def vacuum_f(i_tf_amps, n_turns: int) -> np.ndarray:
    """F = R B_t of the TF coil alone (diamagnetism ignored; a few % at low beta)."""
    return MU0 * n_turns * np.abs(i_tf_amps) / (2 * np.pi)


def q_at(psi, R, Z, f, machine: str, psi_n: float = 0.95, mask_coarse=None, mask_f=None,
         wall_r: float | None = None) -> float:
    """q on the psi_N surface of one (nZ, nR) frame, NaN when the axis or LCFS is not found.

    Pass the scorer's envelope masks: without them the axis search can lock onto a flux maximum
    outside the vessel (seen on MAST next to the solenoid). `wall_r` is an inboard limiter
    radius: the scorer's LCFS finder ignores the wall, so a plasma leaning on the centre column
    gets a boundary inside the column; the boundary flux is then capped at psi(wall_r, Z_axis)."""
    from contour import encloses_point, find_contours, grid_to_physical
    from derive import _bilinear, magnetic_axis
    from lcfs import extract_lcfs
    psi = np.asarray(psi, float)
    lcfs = extract_lcfs(psi, R, Z, machine, mask_coarse, mask_f)
    if lcfs is None:
        return np.nan
    ra, za, iz, ir = magnetic_axis(psi, R, Z, machine, mask_coarse)
    if not np.isfinite(ra):
        return np.nan
    p_ax = float(_bilinear(psi, R, Z, np.array([ra]), np.array([za]))[0])
    p_b = float(np.mean(_bilinear(psi, R, Z, lcfs[:, 0], lcfs[:, 1])))
    if wall_r:
        p_w = float(_bilinear(psi, R, Z, np.array([wall_r]), np.array([za]))[0])
        if abs(p_w - p_ax) < abs(p_b - p_ax):
            p_b = p_w
    level = p_ax + psi_n * (p_b - p_ax)
    # Axis position in fractional grid indices, for the enclosure test in index space.
    row, col = np.interp(za, Z, np.arange(len(Z))), np.interp(ra, R, np.arange(len(R)))
    cands = [c for c in find_contours(psi, level)
             if len(c) > 8 and np.allclose(c[0], c[-1], atol=1e-6) and encloses_point(c, row, col)]
    if not cands:
        return np.nan
    pts = grid_to_physical(min(cands, key=len), R, Z)
    dpz, dpr = np.gradient(psi, Z, R)
    mid = (pts[1:] + pts[:-1]) / 2
    grad = np.hypot(_bilinear(dpr, R, Z, mid[:, 0], mid[:, 1]),
                    _bilinear(dpz, R, Z, mid[:, 0], mid[:, 1]))
    dl = np.hypot(*np.diff(pts, axis=0).T)
    return float(f / (2 * np.pi) * np.sum(dl / (mid[:, 0] * np.maximum(grad, 1e-12))))
