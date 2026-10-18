import numpy as np
import pytest

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


def test_smooth_frames_keeps_length_constants_and_damps_spike():
    from fusion_eq.mast import smooth_frames
    x = np.ones((9, 2, 2))
    x[4] = 6.0
    y = smooth_frames(x, 5)
    assert y.shape == x.shape and np.allclose(y[0], 1.0) and y[4].max() == 2.0
    assert np.array_equal(smooth_frames(x, 1), x)


def test_mast_post_skips_converted_and_constant(tmp_path):
    from fusion_eq.mast_post import convert
    base = dict(psirz=np.zeros((3, 2, 2), np.float32), q95=np.full(3, 7.0, np.float32),
                betaN=np.ones(3, np.float32))
    np.savez(tmp_path / "a.npz", **base, method="freeb", post="v2")
    np.savez(tmp_path / "b.npz", **base, method="const")
    assert convert(None, tmp_path / "a.npz") == "current"
    assert convert(None, tmp_path / "b.npz") == "const"


def test_mast_post_v1_gets_beta_smoothing_once(tmp_path):
    from fusion_eq.mast_post import convert
    beta = np.ones(9, np.float32)
    beta[4] = 10.0
    np.savez(tmp_path / "a.npz", psirz=np.zeros((9, 2, 2), np.float32), q95=np.full(9, 7.0, np.float32),
             betaN=beta, method="freeb", post="v1")
    assert convert(None, tmp_path / "a.npz") == "beta"
    assert convert(None, tmp_path / "a.npz") == "current"
    from fusion_eq.mast import BETA_SMOOTH, smooth_frames
    assert np.allclose(np.load(tmp_path / "a.npz")["betaN"], smooth_frames(beta.astype(float), BETA_SMOOTH))


def test_machine_mask_covers_plasma_region_and_excludes_conductors():
    from fusion_eq.mast import machine_mask
    R, Z = np.linspace(0.06, 2, 65), np.linspace(-2, 2, 65)
    cR = np.r_[np.full(20, 0.14), 0.5, 0.5, 1.1, 1.1, 1.65, 1.65]
    cZ = np.r_[np.linspace(-1.5, 1.5, 20), 1.75, -1.75, 1.1, -1.1, 0.5, -0.5]
    row = {"coil_R": cR, "coil_Z": cZ, "coil_input_column": np.r_[["magnetics_sol_current"] * 20, ["p"] * 6]}
    m = machine_mask(row, R, Z)
    at = lambda r, z: m[np.argmin(abs(Z - z)), np.argmin(abs(R - r))]
    assert at(0.9, 0.0) and at(1.4, 0.3)
    assert not at(0.1, 0.0) and not at(1.9, 0.0) and not at(1.1, 1.1)


def test_pick_sol_scale_falls_back_without_thomson(monkeypatch):
    import fusion_eq.mast as m
    monkeypatch.setattr(m, "thomson_edges", lambda row, t, frac=0.3: (np.full(len(t), np.nan),) * 2)
    row = {"efit_grid_R": np.linspace(0.06, 2, 65), "efit_grid_Z": np.linspace(-2, 2, 65),
           "efit_times": np.arange(10.0)}
    pytest.importorskip("lcfs", reason="needs the vendored scorer on sys.path")
    assert m.pick_sol_scale(row) == m.SOL_SCALE


def test_mast_batch_rows_filters_to_requested_shots(monkeypatch, tmp_path):
    import sys
    import types
    import fusion_eq.mast_batch as b
    fake = types.SimpleNamespace(load_dataset=lambda *a, **k: iter([{"k": i} for i in range(6)]))
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(b, "CACHE", tmp_path)
    np.savez(tmp_path / "shot_0003.npz", sol_scale=0.5)  # a complete cached shot
    assert [i for i, _ in b.rows(0, {1, 3, 4})] == [1, 4]
