"""Write a submission directory that reuses an earlier submission's DIII-D file and rebuilds the
MAST file from per-shot caches, so a MAST-only change does not re-run the DIII-D models.

Caches are overlaid in order: a shot present in a later directory replaces the earlier one.

    PYTHONPATH=src uv run python -m fusion_eq.swap_mast BASE_SUB OUT_SUB CACHE [CACHE ...]
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np

N_MAST = 1206  # public-test MAST shots, in mast_batch order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("caches", type=Path, nargs="+")
    ap.add_argument("--psi-dtype", default="float16", help="float16 as the starter skeleton writes")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d3d = a.out / "diii_d_public_test.npz"
    if not d3d.exists():
        try:
            os.link(a.base / d3d.name, d3d)
        except OSError:
            shutil.copy2(a.base / d3d.name, d3d)
    preds, src = {}, {}
    for i in range(N_MAST):
        f = next((c / f"shot_{i:04d}.npz" for c in reversed(a.caches) if (c / f"shot_{i:04d}.npz").exists()), None)
        if f is None:
            raise FileNotFoundError(f"MAST shot {i} in none of {a.caches}")
        src[f.parent.name] = src.get(f.parent.name, 0) + 1
        d = np.load(f)
        preds[f"shot_{i:04d}_psirz"] = d["psirz"].astype(a.psi_dtype)
        for k in ("q95", "betaN"):
            preds[f"shot_{i:04d}_{k}"] = d[k].astype(np.float32)
    np.savez_compressed(a.out / "mast_public_test.npz", **preds)
    print("shots per cache:", src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
