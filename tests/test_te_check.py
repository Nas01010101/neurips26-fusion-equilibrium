import numpy as np

from fusion_eq.te_check import frame_err


def _grid():
    R = np.linspace(1.0, 2.5, 65); Z = np.linspace(-1.5, 1.5, 65)
    psi = -np.exp(-((R[None] - 1.7) ** 2 + (Z[:, None] / 1.8) ** 2) / 0.2)  # DIII-D sign: axis minimum
    return R, Z, psi


def test_flux_function_te_scores_zero():
    R, Z, psi = _grid()
    pts_R = np.linspace(1.2, 2.2, 20); pts_Z = np.zeros(20)
    te = np.exp(-((pts_R - 1.7) ** 2) / 0.2)  # Te of the flux surface each point sits on
    assert frame_err(psi, R, Z, np.c_[pts_R, pts_Z, te]) < 0.02  # bilinear interp breaks mirrored ties


def test_shifted_axis_is_penalised():
    R, Z, psi = _grid()
    pts_R = np.linspace(1.2, 2.2, 20)
    te = np.exp(-((pts_R - 1.9) ** 2) / 0.2)  # Te peaks 0.2 m outboard of the psi axis
    assert frame_err(psi, R, Z, np.c_[pts_R, np.zeros(20), te]) > 0.1


def test_too_few_points_is_nan():
    R, Z, psi = _grid()
    assert np.isnan(frame_err(psi, R, Z, np.zeros((3, 3))))
