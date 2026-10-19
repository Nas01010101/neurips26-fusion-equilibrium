"""Filesystem layout. Bulk data lives on the SSD; the internal disk stays lean."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "vendor" / "starter"
DATA_DIR = Path(os.environ.get("FUSION_DATA_DIR", "/Volumes/PortableSSD/datasets/fusion-equilibrium"))
RUNS_DIR = Path(os.environ.get("FUSION_RUNS_DIR", "/Volumes/PortableSSD/scratch/fusion-eq-runs"))
HF_REPO = "Sophelio/fusion-equilibrium-challenge"
# Test split to predict: public_test (Development phase) or private_test (Final phase).
SPLIT = os.environ.get("FUSION_SPLIT", "public_test")
# The Final inputs live in their own dataset repo, pinned to the revision the organisers announced.
FINAL_REPO = "ganatma/fusion-equilibrium-challenge-final"
FINAL_REVISION = "965040c12b814f33c8e7209f6b1799932247e793"
DATA_REPO, DATA_REVISION = (FINAL_REPO, FINAL_REVISION) if SPLIT == "private_test" else (HF_REPO, None)


def use_starter() -> Path:
    """Put the vendored starter kit (loaders, errata fixes, official scorer) on sys.path."""
    if not STARTER.exists():
        raise FileNotFoundError(f"{STARTER} missing; run `make vendor`")
    if str(STARTER) not in sys.path:
        sys.path.insert(0, str(STARTER))
    return STARTER


def save_npz(path: Path, **arrays) -> None:
    """Write an .npz atomically and durably: a rename alone survived rtx's power losses with the
    data still unwritten (zero-length shots), so the data and the directory entry are fsynced."""
    import numpy as np
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def npz_done(path: Path) -> bool:
    """True for a complete .npz. Resume checks use this, not exists(), so a file torn by a power
    loss is recomputed instead of crashing the scorer later."""
    import zipfile
    path = Path(path)
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False
