"""Predict the MAST test split (FUSION_SPLIT, default public_test) in parallel into a per-shot
cache that make_submission reads.

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

from .paths import HF_REPO, RUNS_DIR, SPLIT

CACHE = RUNS_DIR / "mast" / "preds"
KW = {}  # predict_mast options (set per worker by _init)
SOL_FROM = None  # cache whose per-shot sol_scale is reused; shots missing there fall back to --sol


def cache_path(i: int) -> Path:
    return CACHE / f"shot_{i:04d}.npz"


def _init(cache=None, kw=None, sol_from=None):
    global CACHE, KW, SOL_FROM
    if cache:
        CACHE = Path(cache)
    KW, SOL_FROM = kw or {}, sol_from
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
    kw = dict(KW)
    prev = Path(SOL_FROM or "") / p.name
    if SOL_FROM and prev.exists() and float(np.load(prev)["sol_scale"]) > 0:  # -1 marks a fallback shot
        kw["sol_scale"] = float(np.load(prev)["sol_scale"])
    try:
        out = predict_mast(row, **kw)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        out = None
    out, how = finite_or_fallback(row, out)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, psirz=out["psirz"].astype(np.float32), q95=out["q95"].astype(np.float32),
             betaN=out["betaN"].astype(np.float32), method=how, sol_scale=out.get("sol_scale", -1.0),
             **({} if how == "const" else {"beta": np.array("total"), "post": np.array("v2")}))  # see mast_rescale, mast_post
    os.replace(tmp, p)  # atomic: a killed run never leaves a half-written shot behind
    return i, how, time.time() - t0


def rows(max_shots: int, only: set | None = None, rows_dir: str | None = None):
    if rows_dir:  # per-shot pickles of the same stream (NNNN.pkl): no re-download per run
        import pickle
        for f in sorted(Path(rows_dir).glob("[0-9][0-9][0-9][0-9].pkl")):
            i = int(f.stem)
            if (not max_shots or i < max_shots) and (only is None or i in only) and not cache_path(i).exists():
                with open(f, "rb") as fh:
                    yield i, pickle.load(fh)
        return
    from datasets import load_dataset
    for i, row in enumerate(load_dataset(HF_REPO, f"mast_{SPLIT}", split=SPLIT, streaming=True)):
        if max_shots and i >= max_shots:
            return
        if (only is None or i in only) and not cache_path(i).exists():
            yield i, row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-shots", type=int, default=0, help="0 = all")
    ap.add_argument("--cache", default=None, help="output directory (default RUNS_DIR/mast/preds)")
    ap.add_argument("--sol", default=None, help="solenoid scale passed to predict_mast, e.g. thomson")
    ap.add_argument("--shots", default=None, help="file of shot indices to run, one per line")
    ap.add_argument("--rows", default=None, help="directory of NNNN.pkl rows to read instead of streaming")
    ap.add_argument("--sol-from", default=None, help="cache to reuse each shot's sol_scale from (else --sol)")
    ap.add_argument("--profile", default="power", choices=("power", "spitzer"))
    ap.add_argument("--vertical", action="store_true", help="in-loop force-balance vertical shift")
    ap.add_argument("--warm", action="store_true", help="search beta0 near the previous frame's")
    a = ap.parse_args()
    kw = {"profile": a.profile, "vertical": a.vertical, "warm": a.warm}
    if a.sol:
        kw["sol_scale"] = a.sol if a.sol == "thomson" else float(a.sol)
    _init(a.cache, kw, a.sol_from)
    only = {int(x) for x in open(a.shots).read().split()} if a.shots else None
    CACHE.mkdir(parents=True, exist_ok=True)
    counts, t0 = {}, time.time()
    stop = os.path.expanduser("~/.claude/STOP")
    with Pool(a.workers, initializer=_init, initargs=(a.cache, kw, a.sol_from)) as pool:
        # Bounded batches: imap would drain the stream eagerly and hold every row (~5 MB) in RAM.
        it, n = rows(a.max_shots, only, a.rows), 0
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
