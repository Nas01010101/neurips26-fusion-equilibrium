"""Axisymmetric Green's functions: poloidal flux of coil filaments and plasma current on the EFIT grid.

Units follow EFIT: psi in Wb/rad (total poloidal flux / 2*pi), currents in A. Each shipped coil
rectangle is treated as a filament at its centre; MAST ships one row per conductor turn, so the
filament approximation is already fine-grained there.
"""
import numpy as np
from scipy.special import ellipe, ellipk

MU0 = 4e-7 * np.pi


def psi_filament(R, Z, Rc, Zc):
    """Flux at (R, Z) per ampere in a circular filament at (Rc, Zc)."""
    k2 = 4 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)
    k2 = np.clip(k2, 1e-12, 1 - 1e-9)
    return MU0 / (2 * np.pi) * np.sqrt(R * Rc) * ((2 - k2) * ellipk(k2) - 2 * ellipe(k2)) / np.sqrt(k2)


def interp_signal(t, x, tq):
    """Linear interpolation that skips NaN samples (MAST early-campaign union time base)."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    ok = np.isfinite(x)
    return np.interp(tq, t[ok], x[ok])


def coil_greens(row, R, Z) -> dict:
    """{current column: (nZ, nR) flux per ampere of that column}.

    Shaping-coil columns carry total ampere-turns spread over the rows that share them, so each
    row gets 1/n of the column; the solenoid column is per-turn current (kA, one row per turn), so
    every row carries the full value. See the starter README units note.
    """
    Rg, Zg = np.meshgrid(R, Z)
    cols = np.asarray(row["coil_input_column"])
    cR, cZ = np.asarray(row["coil_R"], float), np.asarray(row["coil_Z"], float)
    out = {}
    for c in np.unique(cols):
        m = cols == c
        g = sum(psi_filament(Rg, Zg, a, b) for a, b in zip(cR[m], cZ[m]))
        out[c] = g if c.endswith("sol_current") else g / m.sum()
    return out


def vacuum_psi(row, R, Z, times, scale: dict | None = None) -> np.ndarray:
    """(T, nZ, nR) coil-driven flux at `times` (ms). `scale` multiplies selected columns."""
    scale = scale or {}
    tm = row["magnetics_time"]
    out = np.zeros((len(times), len(Z), len(R)))
    for c, g in coil_greens(row, R, Z).items():
        amps = interp_signal(tm, row[c], times) * 1e3 * scale.get(c, 1.0)
        out += amps[:, None, None] * g[None]
    return out


_GRID_CACHE: dict = {}


def grid_greens(R, Z) -> np.ndarray:
    """(nZ*nR, nZ*nR) flux at every grid node per ampere at every node. ~140 MB for 65x65, built
    once per grid. The singular self-term is replaced by the mean of its 4 neighbours' couplings,
    a finite stand-in for a filament of the cell's own size."""
    key = (R.tobytes(), Z.tobytes())
    if key not in _GRID_CACHE:
        Rg, Zg = np.meshgrid(R, Z)
        r, z = Rg.ravel(), Zg.ravel()
        G = psi_filament(r[:, None], z[:, None], r[None, :], z[None, :])
        nZ, nR = Rg.shape
        dR, dZ = R[1] - R[0], Z[1] - Z[0]
        for i in range(r.size):
            G[i, i] = np.mean([psi_filament(r[i] + a, z[i] + b, r[i], z[i])
                               for a, b in ((dR / 2, 0), (-dR / 2, 0), (0, dZ / 2), (0, -dZ / 2))])
        _GRID_CACHE[key] = G
    return _GRID_CACHE[key]


def plasma_psi(J, R, Z, ip) -> np.ndarray:
    """Flux (nZ, nR) of toroidal current density J (nZ, nR, any scale) normalised to total ip (A)."""
    s = J.sum()
    if s == 0:
        return np.zeros_like(J, dtype=float)
    return (grid_greens(R, Z) @ (J.ravel() / s)).reshape(J.shape) * ip
