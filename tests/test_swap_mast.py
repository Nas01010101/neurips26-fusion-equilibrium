import sys

import numpy as np

from fusion_eq import swap_mast


def _shot(d, i, v):
    d.mkdir(exist_ok=True)
    np.savez(d / f"shot_{i:04d}.npz", psirz=np.full((2, 3, 3), v, np.float32),
             q95=np.full(2, v, np.float32), betaN=np.full(2, v, np.float32))


def test_later_cache_overlays_earlier_and_keeps_base_diiid(tmp_path, monkeypatch):
    base, out, a, b = (tmp_path / n for n in ("base", "out", "a", "b"))
    base.mkdir()
    (base / f"diii_d_{swap_mast.SPLIT}.npz").write_bytes(b"d3d")
    _shot(a, 0, 1.0); _shot(a, 1, 1.0); _shot(b, 1, 2.0)
    monkeypatch.setattr(sys, "argv", ["x", str(base), str(out), str(a), str(b), "--n-shots", "2"])
    assert swap_mast.main() == 0
    m = np.load(out / f"mast_{swap_mast.SPLIT}.npz")
    assert m["shot_0000_q95"][0] == 1.0 and m["shot_0001_q95"][0] == 2.0
    assert m["shot_0001_psirz"].dtype == np.float16
    assert (out / f"diii_d_{swap_mast.SPLIT}.npz").read_bytes() == b"d3d"


def test_shot_counts_cover_both_splits():
    assert swap_mast.N_MAST == {"public_test": 1206, "private_test": 1205}
