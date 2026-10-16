"""Vertical plasma position from coil force balance, applied to a solved cache without re-solving.

freeb.solve symmetrises J about Z = 0 to keep the Picard loop stable, so every predicted magnetic
axis sits within millimetres of the midplane. On MAST the vertical position is held by feedback
on the radial-field coils, which keeps the plasma where the coils' net vertical force on its
current vanishes: F_z = -2 pi int R J B_R dA with R B_R = -d psi_vac / dZ. The plasma's own field
exerts no net force on it, so only the vacuum flux enters. We find the rigid shift z of the solved
current that zeroes F_z and move the plasma flux by the same z (the Green's function is invariant
under Z translation, so the shifted flux is exact for the shifted current).

Uses MAST inputs only (coil currents, solved psi); no label enters.

    PYTHONPATH=src uv run python -m fusion_eq.mast_vertical SRC_DIR DST_DIR
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, shift

from .greens import vacuum_psi
from .mast import SOL_SCALE, machine_mask

# Search range for the shift. A larger zero of F_z means no nearby equilibrium was found; the
# vessel allows more, but controlled MAST plasmas sit near the midplane: the vertical growth rate
# climbs from 40/s below 0.1 m to 80/s by 0.25 m and diverges past ~0.4 m (Windridge et al.,
# PPCF 53, 035018 (2011)), and 1-2 cm axis shifts already flip H-mode access (O'Gorman et al.,
# Nucl. Fusion 54, 082002 (2014)). 0.15 m bounds that range; machine physics, not fitted.
Z_MAX = 0.15
Z_GRID = np.linspace(-0.3, 0.3, 61)
WIN = 5  # frames per running median of z: one bad current estimate should not move a frame


def plasma_current(psip, R, Z, mask) -> np.ndarray:
    """J_phi (nZ, nR, arbitrary positive scale) from the plasma flux by the Grad-Shafranov
    operator, -mu0 R J = R d/dR (1/R dpsi/dR) + d2psi/dZ2, kept where it carries the plasma sign
    and is above 5% of its peak (finite differences ring outside the current channel)."""
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    lap = (R[None] * np.gradient(np.gradient(psip, dR, axis=1) / R[None], dR, axis=1)
           + np.gradient(np.gradient(psip, dZ, axis=0), dZ, axis=0))
    J = np.where(mask, -lap / R[None], 0.0)
    J *= np.sign(J.sum()) or 1.0
    J = np.maximum(J, 0)
    return np.where(J > 0.05 * J.max(), J, 0.0) if J.max() > 0 else J


def force_zero(J, dvac_dz, dZ) -> float:
    """Rigid shift z (m) of J at which int J(R, Z - z) dpsi_vac/dZ dA = 0, nearest to the
    midplane; NaN when there is none within Z_GRID."""
    G = np.array([(shift(J, (z / dZ, 0), order=1, mode="constant") * dvac_dz).sum() for z in Z_GRID])
    s = np.flatnonzero(np.sign(G[:-1]) != np.sign(G[1:]))
    if not s.size:
        return np.nan
    k = s[np.argmin(np.abs(Z_GRID[s]))]
    return float(Z_GRID[k] - G[k] * (Z_GRID[k + 1] - Z_GRID[k]) / (G[k + 1] - G[k]))


def vertical_shifts(psi, vac, R, Z, mask) -> np.ndarray:
    """Per-frame z (m): force-balance zero, clipped to +-Z_MAX, NaN frames 0, running median."""
    dZ = Z[1] - Z[0]
    z = np.array([force_zero(plasma_current(psi[k] - vac[k], R, Z, mask),
                             np.gradient(vac[k], dZ, axis=0), dZ) for k in range(len(psi))])
    z = np.clip(np.nan_to_num(z, nan=0.0), -Z_MAX, Z_MAX)
    return median_filter(z, size=WIN, mode="nearest")


def apply(row, d: dict, z: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """Shifted copy of a cache entry and the per-frame z (computed unless given). Constant
    fallbacks are returned as is."""
    if str(d["method"]) == "const":
        return d, np.zeros(len(d["q95"]))
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    s = float(d.get("sol_scale", -1.0))
    vac = vacuum_psi(row, R, Z, np.asarray(row["efit_times"], float),
                     {"magnetics_sol_current": s if s > 0 else SOL_SCALE})
    psi = d["psirz"].astype(float)
    if z is None:
        z = vertical_shifts(psi, vac, R, Z, machine_mask(row, R, Z))
    dZ = Z[1] - Z[0]
    out = np.stack([vac[k] + shift(psi[k] - vac[k], (z[k] / dZ, 0), order=3, mode="nearest")
                    for k in range(len(psi))])
    return {**d, "psirz": out.astype(np.float32), "z_shift": z.astype(np.float32)}, z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rows", default="/Volumes/PortableSSD/scratch/fusion-eq-runs/mast/rows")
    ap.add_argument("--z-from", help="reuse z_shift stored in this shifted cache instead of solving")
    a = ap.parse_args()
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.src).glob("shot_*.npz")):
        if (dst / f.name).exists():
            continue
        row = pickle.load(open(Path(a.rows) / f"{f.stem[5:]}.pkl", "rb"))
        z = None
        if a.z_from:
            prev = np.load(Path(a.z_from) / f.name)
            z = prev["z_shift"] if "z_shift" in prev else None
            if z is None:  # constant fallback: apply() leaves it unshifted anyway
                z = np.zeros(len(prev["q95"]))
        d, z = apply(row, dict(np.load(f)), z)
        tmp = dst / (f.stem + ".tmp.npz")
        np.savez(tmp, **d)
        os.replace(tmp, dst / f.name)
        print(f.stem, f"z mean {z.mean():+.3f} min {z.min():+.3f} max {z.max():+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
