import numpy as np

from fusion_eq import d3d_bench


def test_tsmooth_keeps_coil_flux_and_steady_plasma(monkeypatch):
    rng = np.random.default_rng(0)
    vac = rng.normal(size=(6, 5, 4))
    plasma = rng.normal(size=(5, 4))
    monkeypatch.setattr(d3d_bench, "vacuum", lambda row, frames: (None, None, None, vac, None, -1.0, None))
    p = -(-1.0 * vac + plasma)  # stored sign of solver psi s*vac + plasma
    np.testing.assert_allclose(d3d_bench.tsmooth({}, np.arange(6), p, 1.0), p, atol=1e-12)
    # a one-frame plasma spike is spread out, the coil flux is not
    spiked = p.copy()
    spiked[3] -= 1.0
    out = d3d_bench.tsmooth({}, np.arange(6), spiked, 1.0)
    assert np.abs(out[3] - p[3]).max() < 0.5 and np.abs(out[2] - p[2]).max() > 0.1
