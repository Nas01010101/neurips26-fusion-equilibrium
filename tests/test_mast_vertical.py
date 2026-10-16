import numpy as np

from fusion_eq.greens import plasma_psi, psi_filament
from fusion_eq.mast_vertical import force_zero, plasma_current

R = np.linspace(0.06, 2.0, 65)
Z = np.linspace(-2.0, 2.0, 65)
Rg, Zg = np.meshgrid(R, Z)


def blob(z0):
    return np.exp(-((Rg - 0.9) ** 2 + (Zg - z0) ** 2 / 1.8 ** 2) / 0.1)


def test_force_zero_finds_the_centre_of_a_shifted_symmetric_field():
    # Shaping coils mirror-symmetric about Z = h: the force on the current vanishes at shift h.
    h = 0.06
    vac = sum(1e5 * psi_filament(Rg, Zg, r, h + s * dz)
              for r, dz in ((1.5, 1.0), (0.5, 1.6)) for s in (1, -1))
    z = force_zero(blob(0.0), np.gradient(vac, Z[1] - Z[0], axis=0), Z[1] - Z[0])
    assert abs(z - h) < 0.01


def test_force_zero_is_midplane_for_symmetric_coils():
    vac = sum(1e5 * psi_filament(Rg, Zg, 1.5, s) for s in (1.0, -1.0))
    assert abs(force_zero(blob(0.0), np.gradient(vac, Z[1] - Z[0], axis=0), Z[1] - Z[0])) < 0.01


def test_plasma_current_recovers_the_current_centroid():
    J = blob(0.25)
    Jr = plasma_current(plasma_psi(J, R, Z, 5e5), R, Z, np.ones_like(J, bool))
    assert abs((Jr * Zg).sum() / Jr.sum() - 0.25) < 0.03
