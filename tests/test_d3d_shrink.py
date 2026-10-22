import sys

import numpy as np

from fusion_eq import d3d_shrink


def test_shrinks_scalars_only(tmp_path, monkeypatch):
    psi = np.random.default_rng(0).normal(size=(3, 65, 65)).astype(np.float32)
    q = np.array([3.0, 5.0, 9.0], np.float32)
    b = np.array([0.5, 1.0, 3.0], np.float32)
    src, out = tmp_path / "in.npz", tmp_path / "out.npz"
    np.savez_compressed(src, shot_0000_psirz=psi, shot_0000_q95=q, shot_0000_betaN=b)
    monkeypatch.setattr(sys, "argv", ["x", str(src), str(out), "--lam", "0.5"])
    assert d3d_shrink.main() == 0
    z = np.load(out)
    assert np.array_equal(z["shot_0000_psirz"], psi)
    m = d3d_shrink.TRAIN_MEAN
    np.testing.assert_allclose(z["shot_0000_q95"], m["q95"] + 0.5 * (q - m["q95"]), rtol=1e-6)
    np.testing.assert_allclose(z["shot_0000_betaN"], m["betaN"] + 0.5 * (b - m["betaN"]), rtol=1e-6)
    assert z["shot_0000_q95"].dtype == np.float32
