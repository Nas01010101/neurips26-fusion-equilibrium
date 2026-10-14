"""Filesystem layout. Bulk data lives on the SSD; the internal disk stays lean."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "vendor" / "starter"
DATA_DIR = Path(os.environ.get("FUSION_DATA_DIR", "/Volumes/PortableSSD/datasets/fusion-equilibrium"))
RUNS_DIR = Path(os.environ.get("FUSION_RUNS_DIR", "/Volumes/PortableSSD/scratch/fusion-eq-runs"))
HF_REPO = "Sophelio/fusion-equilibrium-challenge"


def use_starter() -> Path:
    """Put the vendored starter kit (loaders, errata fixes, official scorer) on sys.path."""
    if not STARTER.exists():
        raise FileNotFoundError(f"{STARTER} missing; run `make vendor`")
    if str(STARTER) not in sys.path:
        sys.path.insert(0, str(STARTER))
    return STARTER
