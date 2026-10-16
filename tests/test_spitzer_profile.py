import numpy as np

from fusion_eq.mast import spitzer_profile

R = np.linspace(0.2, 1.6, 65)
Z = np.linspace(-1.0, 1.0, 65)
Rg, Zg = np.meshgrid(R, Z)
PN = np.clip(((Rg - 0.9) ** 2 + (Zg / 1.8) ** 2) / 0.6 ** 2, 0, 1)  # circular-ish surfaces, a = 0.6
INSIDE = PN < 1
R_CH = np.linspace(0.25, 1.55, 130)


def te_on(pn_ch):
    return np.where(pn_ch < 1, 1000 * (1 - pn_ch) + 20, 5.0)


def test_profile_is_te_to_the_three_halves_and_vanishes_at_the_lcfs():
    pn_ch = np.clip((R_CH - 0.9) ** 2 / 0.36, 0, 1)
    f = spitzer_profile(R_CH, te_on(pn_ch), R, Z)(PN, INSIDE)
    iz = np.argmin(np.abs(Z))
    core = np.argmin(np.abs(R - 0.9))
    assert abs(f[iz, core] - 1) < 0.05
    assert f[PN >= 1].max() < 1e-9
    mid = np.argmin(np.abs(R - 0.9 - 0.6 * np.sqrt(0.5)))  # pn = 0.5 on the midplane
    assert abs(f[iz, mid] - 0.5 ** 1.5) < 0.1
    assert np.all(np.diff(f[iz, core:]) <= 1e-12)  # non-increasing outward


def test_spiky_channel_cannot_make_the_profile_hollow():
    pn_ch = np.clip((R_CH - 0.9) ** 2 / 0.36, 0, 1)
    te = te_on(pn_ch)
    te[np.argmin(np.abs(R_CH - 1.3))] = 1e5
    f = spitzer_profile(R_CH, te, R, Z)(PN, INSIDE)
    assert f.max() <= 1 + 1e-9


def test_falls_back_to_linear_shape_without_thomson():
    f = spitzer_profile(R_CH, np.full(130, np.nan), R, Z)(PN, INSIDE)
    assert np.allclose(f, 1 - PN)
