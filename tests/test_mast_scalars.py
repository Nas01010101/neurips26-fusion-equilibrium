import numpy as np

from fusion_eq.mast import q95_iter, robust_series


def test_q95_iter_matches_iter_baseline():
    # ITER 15 MA baseline: R 6.2, a 2.0, B 5.3 T, kappa95 1.7, delta95 0.33 -> q95 ~ 3.
    assert abs(q95_iter(6.2, 2.0, 5.3, 15.0, 1.7, 0.33) - 3.0) < 0.3


def test_robust_series_removes_spike_and_out_of_range():
    x = np.array([6.0, 6.1, 30.0, 6.2, np.nan, 6.3, 6.4])
    y = robust_series(x, 2, 25)
    assert np.isfinite(y).all() and y.max() < 7


def test_robust_series_all_invalid_stays_nan():
    assert np.isnan(robust_series(np.array([np.nan, 99.0]), 2, 25)).all()
