"""Apply `mast.finalize` (time smoothing, wall-limited q95) to a cache that mast_batch wrote
before finalize existed, without re-solving any equilibrium.

The vacuum flux is recomputed from the coil currents, so only the plasma part is smoothed.
Converted files carry `post="v1"`; files that mast_batch writes now are born with it, so nothing
is smoothed twice. Constant fallbacks are left alone.

    PYTHONPATH=src uv run python -m fusion_eq.mast_post [--max-shots N]
"""
import argparse
import os

import numpy as np

from .greens import vacuum_psi
from .mast import SOL_SCALE, finalize
from .mast_batch import cache_path

POST = "v1"


def convert(row, path) -> str:
    d = dict(np.load(path))
    if "post" in d:
        return "current"
    if str(d["method"]) == "const":
        return "const"
    vac = vacuum_psi(row, np.asarray(row["efit_grid_R"], float), np.asarray(row["efit_grid_Z"], float),
                     np.asarray(row["efit_times"], float), {"magnetics_sol_current": SOL_SCALE})
    psi, q95 = finalize(row, d["psirz"].astype(float), vac, d["q95"].astype(float))
    d.update(psirz=psi.astype(np.float32), q95=q95.astype(np.float32), post=np.array(POST))
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **d)
    os.replace(tmp, path)  # atomic, as in mast_batch
    return "converted"


def main() -> int:
    from datasets import load_dataset
    from .paths import HF_REPO, use_starter
    use_starter()
    import local_score  # noqa: F401  puts fusion_scoring/ on sys.path
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shots", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    counts = {}
    for i, row in enumerate(load_dataset(HF_REPO, "mast_public_test", split="public_test", streaming=True)):
        if a.max_shots and i >= a.max_shots:
            break
        how = convert(row, cache_path(i)) if cache_path(i).exists() else "missing"
        counts[how] = counts.get(how, 0) + 1
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
