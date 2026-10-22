import numpy as np

from fusion_eq import twin


def test_fit_learns_a_linear_residual_and_delta_is_offset_free(tmp_path, monkeypatch):
    monkeypatch.setattr(twin, "K_PHI", 4)
    monkeypatch.setattr(twin, "K_VAC", 2)
    rng = np.random.default_rng(0)
    n, m = 400, 30
    modes = rng.normal(size=(4, m))
    for s in range(10):  # 10 "shots"; truth = pipeline + a mode set by the first scalar
        c = rng.normal(size=(40, 4))
        sc = rng.normal(size=(40, 4)).astype(np.float32)
        ph = c @ modes + 5.0  # an offset the twin must ignore
        pt = ph + np.outer(sc[:, 0], modes[1])
        np.savez(tmp_path / f"{s}.npz", phi_t=pt.astype(np.float32), phi_h=ph.astype(np.float32),
                 vacn=rng.normal(size=(40, m)).astype(np.float32), scal=sc)
    tw = twin.fit(tmp_path, tmp_path / "m.pkl", kind="ridge")
    pt, ph, vn, sc, _ = twin.load_data(tmp_path)
    d = tw.delta(ph, vn, sc)
    np.testing.assert_allclose(d.mean(1), 0, atol=1e-9)
    r = pt - ph
    r -= r.mean(1, keepdims=True)
    assert ((r - d) ** 2).sum() < 0.01 * (r ** 2).sum()
