"""Predict the MAST public test set in parallel into a per-shot cache that make_submission reads.

The free-boundary solve costs ~1.6 s per frame, so the 1206-shot set is hours of CPU: shots run
in a process pool, each result lands in its own file, and a rerun skips files already written.
Shot k is the k-th row of the streamed split, the same order submission_skeleton enumerates.

    PYTHONPATH=src uv run python -m fusion_eq.mast_batch --workers 8 [--max-shots 5]
"""
import argparse
import os
import sys
import time
import traceback
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .paths import HF_REPO, RUNS_DIR

CACHE = RUNS_DIR / "mast" / "preds"


def cache_path(i: int) -> Path:
    return CACHE / f"shot_{i:04d}.npz"


def _init():
    # Each worker needs the starter (data_fixes) and the scorer modules (lcfs, derive) importable.
    from .paths import use_starter
    use_starter()
    import local_score  # noqa: F401  puts fusion_scoring/ on sys.path


def finite_or_fallback(row, out: dict | None) -> tuple[dict, str]:
    """The validator rejects any non-finite value. Fall back to the fixed-shape Thomson model,
    then to vacuum-only flux with constant scalars, rather than lose the shot."""
    from .mast import predict_mast
    ok = lambda o: o is not None and all(np.isfinite(o[k]).all() for k in ("psirz", "q95", "betaN"))  # noqa: E731
    if ok(out):
        return out, "freeb"
    try:
        out = predict_mast(row, freeb_iter=0)
        if ok(out):
            return out, "thomson"
    except Exception:
        pass
    T = len(np.asarray(row["efit_times"]))
    psi = np.nan_to_num(np.asarray(out["psirz"], np.float32)) if out is not None else np.zeros((T, 65, 65), np.float32)
    return {"psirz": psi, "q95": np.full(T, 7.0, np.float32), "betaN": np.full(T, 1.0, np.float32)}, "const"


def _work(args):
    i, row = args
    p = cache_path(i)
    if p.exists():
        return i, "cached", 0.0
    from .mast import predict_mast
    t0 = time.time()
    try:
        out = predict_mast(row)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        out = None
    out, how = finite_or_fallback(row, out)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, psirz=out["psirz"].astype(np.float32), q95=out["q95"].astype(np.float32),
             betaN=out["betaN"].astype(np.float32), method=how,
             **({} if how == "const" else {"beta": np.array("total"), "post": np.array("v2")}))  # see mast_rescale, mast_post
    os.replace(tmp, p)  # atomic: a killed run never leaves a half-written shot behind
    return i, how, time.time() - t0


def rows(max_shots: int):
    from datasets import load_dataset
    for i, row in enumerate(load_dataset(HF_REPO, "mast_public_test", split="public_test", streaming=True)):
        if max_shots and i >= max_shots:
            return
        if not cache_path(i).exists():
            yield i, row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-shots", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    counts, t0 = {}, time.time()
    stop = os.path.expanduser("~/.claude/STOP")
    with Pool(a.workers, initializer=_init) as pool:
        # Bounded batches: imap would drain the stream eagerly and hold every row (~5 MB) in RAM.
        it, n = rows(a.max_shots), 0
        while batch := list(islice(it, 2 * a.workers)):
            for i, how, dt in pool.imap_unordered(_work, batch):
                n += 1
                counts[how] = counts.get(how, 0) + 1
                if n % 10 == 0 or how != "freeb":
                    print(f"{time.strftime('%H:%M')} shot {i:04d} {how} {dt:.0f}s  done {n}  {counts}  "
                          f"elapsed {(time.time() - t0) / 60:.0f} min", flush=True)
            if os.path.exists(stop):
                print("STOP file present; exiting after this batch", flush=True)
                break
    print(f"finished: {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
