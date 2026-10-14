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


def test_total_beta_n_scales_and_despikes():
    from fusion_eq.mast import total_beta_n
    x = np.full(9, 0.65)
    x[4] = 5.0  # single-frame Thomson spike
    out = total_beta_n(x, 0.35)
    assert np.allclose(out, 1.0)


def test_total_beta_n_keeps_input_when_nothing_valid():
    from fusion_eq.mast import total_beta_n
    x = np.full(5, 50.0)  # outside the physical range everywhere
    assert np.array_equal(total_beta_n(x, 0.35), x)


def test_convert_cache_rescales_once(tmp_path):
    from fusion_eq.mast_rescale import convert_cache
    np.savez(tmp_path / "shot_0000.npz", psirz=np.zeros((5, 2, 2), np.float32),
             q95=np.full(5, 7.0, np.float32), betaN=np.full(5, 0.65, np.float32), method="freeb")
    np.savez(tmp_path / "shot_0001.npz", psirz=np.zeros((5, 2, 2), np.float32),
             q95=np.full(5, 7.0, np.float32), betaN=np.full(5, 1.0, np.float32), method="const")
    assert convert_cache(tmp_path) == {"converted": 1, "current": 0, "const": 1}
    assert convert_cache(tmp_path) == {"converted": 0, "current": 1, "const": 1}
    assert np.allclose(np.load(tmp_path / "shot_0000.npz")["betaN"], 1.0)
    assert np.allclose(np.load(tmp_path / "shot_0001.npz")["betaN"], 1.0)
