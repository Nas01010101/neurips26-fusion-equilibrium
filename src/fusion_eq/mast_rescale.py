"""Bring the MAST cache up to date with `mast.total_beta_n` without re-solving any equilibrium.

betaN depends only on Thomson and the coil currents, not on psi, so caches written before the
fast-ion correction can be converted in place. Converted files carry `beta="total"`; files that
mast_batch writes now are born with it, so nothing is converted twice.

    PYTHONPATH=src uv run python -m fusion_eq.mast_rescale              # convert the cache
    PYTHONPATH=src uv run python -m fusion_eq.mast_rescale --npz IN OUT # rewrite a built npz
"""
import argparse
import os
from pathlib import Path

import numpy as np

from .mast import total_beta_n
from .mast_batch import CACHE


def convert_cache(cache: Path = CACHE) -> dict:
    counts = {"converted": 0, "current": 0, "const": 0}
    for p in sorted(cache.glob("shot_*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        d = dict(np.load(p))
        if "beta" in d:
            counts["current"] += 1
            continue
        if str(d["method"]) == "const":  # constant fallback is not a thermal estimate
            counts["const"] += 1
            continue
        d["betaN"] = total_beta_n(d["betaN"]).astype(np.float32)
        d["beta"] = np.array("total")
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **d)
        os.replace(tmp, p)
        counts["converted"] += 1
    return counts


def rewrite_npz(src: Path, dst: Path) -> int:
    """Same arrays as `src` except every shot_XXXX_betaN, taken from the converted cache."""
    d = np.load(src)
    out = {}
    for k in d.files:
        if k.endswith("_betaN"):
            c = np.load(CACHE / f"{k[:-len('_betaN')]}.npz")
            if "beta" not in c or len(c["betaN"]) != len(d[k]):
                raise ValueError(f"{k}: cache not converted or frame count differs")
            out[k] = c["betaN"].astype(np.float32)
        else:
            out[k] = d[k]
    np.savez_compressed(dst, **out)  # same container as submission_skeleton
    return sum(k.endswith("_betaN") for k in out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs=2, type=Path, metavar=("IN", "OUT"))
    a = ap.parse_args()
    if a.npz:
        print(f"rewrote betaN for {rewrite_npz(*a.npz)} shots -> {a.npz[1]}")
    else:
        print(convert_cache())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
