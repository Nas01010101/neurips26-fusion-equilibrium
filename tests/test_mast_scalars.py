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
    np.savez(tmp_path / "a.npz", **base, method="freeb", post="v1")
    np.savez(tmp_path / "b.npz", **base, method="const")
    assert convert(None, tmp_path / "a.npz") == "current"
    assert convert(None, tmp_path / "b.npz") == "const"


def test_wall_limited_q95_matches_efit_on_limited_demo_frames():
    # Shot 28350 leans on the centre column for its first frames: the scorer's LCFS then passes
    # inside the column and q95 comes out ~1.5x EFIT unless the boundary is capped at the wall.
    try:
        from fusion_eq.mast_score import load_demo
        s = load_demo("mast_shot_28350")
    except Exception as e:  # demo data not downloaded
        pytest.skip(str(e))
    from fusion_eq.greens import interp_signal
    from fusion_eq.mast import N_TF, WALL_R
    from fusion_eq.paths import STARTER, use_starter
    from fusion_eq.qprofile import q_at, vacuum_f
    use_starter()
    import local_score  # noqa: F401
    m = np.load(STARTER / "fusion_scoring" / "masks" / "mast_envelope.npz")
    mc = m["mask_coarse"].astype(bool)
    t = np.asarray(s["row"]["efit_times"], float)
    f = vacuum_f(interp_signal(s["row"]["magnetics_time"], s["row"]["magnetics_tf_current"], t) * 1e3, N_TF)
    args = (m["grid_R"], m["grid_Z"])
    q = lambda k, w: q_at(s["psi"][k], *args, f[k], "MAST", mask_coarse=mc, mask_f=mc.astype(float), wall_r=w)
    assert q(0, None) > 1.4 * s["q95"][0]
    for k in (0, 3, 6, 20):
        assert abs(q(k, WALL_R) / s["q95"][k] - 1) < 0.1
