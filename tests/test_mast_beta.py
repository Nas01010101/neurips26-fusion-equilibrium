import pickle

import numpy as np

from fusion_eq.mast_beta import recompute


def _shot(path, method="freeb", **extra):
    np.savez(path, psirz=np.zeros((4, 2, 2), np.float32), q95=np.full(4, 7.0, np.float32),
             betaN=np.full(4, 0.5, np.float32), method=method, **extra)


def test_recompute_sets_window_once_and_skips_const(tmp_path):
    cache, rows = tmp_path / "cache", tmp_path / "rows"
    cache.mkdir(), rows.mkdir()
    _shot(cache / "shot_0000.npz")
    _shot(cache / "shot_0001.npz", method="const")
    _shot(cache / "shot_0002.npz", beta_win=np.array(5))
    for i in range(3):
        pickle.dump({"efit_times": np.arange(4.0)}, open(rows / f"{i:04d}.pkl", "wb"))
    calls = []

    def fn(row, win):
        calls.append(win)
        return np.full(len(row["efit_times"]), 1.25)

    assert recompute(cache, rows, 5, fn) == {"recomputed": 1, "const": 1, "current": 1}
    d = np.load(cache / "shot_0000.npz")
    assert int(d["beta_win"]) == 5 and np.allclose(d["betaN"], 1.25) and np.allclose(d["q95"], 7.0)
    assert np.allclose(np.load(cache / "shot_0001.npz")["betaN"], 0.5)
    assert recompute(cache, rows, 5, fn) == {"current": 2, "const": 1} and calls == [5]
    assert not list(cache.glob("*.tmp.npz"))
