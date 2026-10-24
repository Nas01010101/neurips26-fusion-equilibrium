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


def test_mode_scale_weights_each_mode_and_map_follows_overlap(tmp_path):
    rng = np.random.default_rng(1)
    basis = np.linalg.qr(rng.normal(size=(30, 3)))[0].T
    tw = twin.Twin(basis, np.zeros((1, 30)), 0.0, 1.0, None)

    class Reg:
        def predict(self, x):
            return np.ones((len(x), 3))
    tw.reg = Reg()
    tw.x = lambda ph, vn, sc: np.zeros((len(ph), 1))
    one = tw.delta(np.zeros((1, 30)), None, None)
    tw.mode_scale = np.array([1.0, 0.0, 0.5])
    np.testing.assert_allclose(tw.delta(np.zeros((1, 30)), None, None), (np.array([[1.0, 0.0, 0.5]]) @ basis))
    assert not np.allclose(one, tw.delta(np.zeros((1, 30)), None, None))
    # target basis = source modes 2, 0, 1: the mapped scales follow the matching source mode
    import pickle
    del tw.x, tw.reg
    pickle.dump(tw, open(tmp_path / "a.pkl", "wb"))
    tgt = twin.Twin(basis[[2, 0, 1]], np.zeros((1, 30)), 0.0, 1.0, None)
    pickle.dump(tgt, open(tmp_path / "t.pkl", "wb"))
    np.testing.assert_allclose(twin.map_scale([tmp_path / "a.pkl"], tmp_path / "t.pkl", tmp_path / "o.pkl"),
                               [0.5, 1.0, 0.0], atol=1e-12)


def test_blend_scale_pulls_mode_scales_toward_c(tmp_path):
    import pickle
    tw = twin.Twin(np.eye(3), np.zeros((1, 3)), 0.0, 1.0, None)
    tw.mode_scale = np.array([1.0, 0.0, 0.5])
    pickle.dump(tw, open(tmp_path / "a.pkl", "wb"))
    want = [0.8475, 0.0975, 0.4725]  # 0.75 * s + 0.25 * 0.39
    np.testing.assert_allclose(twin.blend_scale(tmp_path / "a.pkl", 0.75, 0.39, tmp_path / "b.pkl"), want)
    np.testing.assert_allclose(pickle.load(open(tmp_path / "b.pkl", "rb")).mode_scale, want)
    np.testing.assert_allclose(twin.blend_scale(tmp_path / "a.pkl", 0.0, 0.39, tmp_path / "c.pkl"), [0.39] * 3)
