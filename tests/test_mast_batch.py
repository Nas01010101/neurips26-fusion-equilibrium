import pickle

import numpy as np

from fusion_eq import mast_batch


def test_rows_dir_skips_cached_and_filters(tmp_path, monkeypatch):
    rows_dir, cache = tmp_path / "rows", tmp_path / "cache"
    rows_dir.mkdir(); cache.mkdir()
    for i in range(4):
        (rows_dir / f"{i:04d}.pkl").write_bytes(pickle.dumps({"i": i}))
    monkeypatch.setattr(mast_batch, "CACHE", cache)
    np.savez(cache / "shot_0001.npz", sol_scale=0.5)
    assert [i for i, _ in mast_batch.rows(0, None, str(rows_dir))] == [0, 2, 3]
    assert [r["i"] for _, r in mast_batch.rows(0, {2, 3}, str(rows_dir))] == [2, 3]
    assert [i for i, _ in mast_batch.rows(3, None, str(rows_dir))] == [0, 2]
