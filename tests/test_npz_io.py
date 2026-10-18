import numpy as np

from fusion_eq.paths import npz_done, save_npz


def test_save_npz_round_trip(tmp_path):
    p = tmp_path / "s.npz"
    save_npz(p, psirz=np.ones((2, 3), np.float32))
    assert npz_done(p)
    assert np.load(p)["psirz"].shape == (2, 3)
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_npz_done_rejects_power_loss_leftovers(tmp_path):
    # rtx power loss left zero-length files behind a completed rename; resume must redo them
    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    torn = tmp_path / "torn.npz"
    save_npz(torn, a=np.arange(1000))
    torn.write_bytes(torn.read_bytes()[:50])
    assert not npz_done(empty)
    assert not npz_done(torn)
    assert not npz_done(tmp_path / "missing.npz")
