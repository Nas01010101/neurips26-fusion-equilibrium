"""Train a registry model on cached folds and score held-out shots with the official scorer.

    PYTHONPATH=src uv run python -m fusion_eq.evaluate --model pca_ridge --n-val 20

The scoring functions are imported from the vendored `local_score.py`, so the numbers use the
same code as Codabench. Reference contours and derived scalars depend only on the ground truth,
so they are cached per shot and every later experiment pays only for scoring its own predictions.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from .data import fold_of, load_packed
from .models import REGISTRY, Ensemble
from .paths import RUNS_DIR, use_starter

try:  # registers the torch models; the sklearn ones work without it
    from . import models_torch  # noqa: F401
except ImportError:
    pass

VAL_FOLD = 4


def _scorer():
    use_starter()
    import local_score as ls
    mask = np.load(ls.HERE / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    mc = mask["mask_coarse"].astype(bool)
    return ls, mask["grid_R"], mask["grid_Z"], mc, mc.astype(np.float64)


def val_shots(cache_dir: Path, n: int) -> list[dict]:
    """First `n` cached shots of the validation fold, all frames kept (the scorer needs them)."""
    out = []
    for f in sorted(cache_dir.glob("*.npz")):
        if fold_of(f.stem + ".parquet") != VAL_FOLD:  # same hash the cache used; no file read
            continue
        z = np.load(f)
        out.append({"name": f.stem, "X": np.nan_to_num(z["X"]), "psi": z["psi"],
                    "q95": z["y"][:, 0].astype(np.float64), "betaN": z["y"][:, 1].astype(np.float64)})
        if len(out) >= n:
            break
    return out


def references(shots, ref_dir: Path):
    ls, R, Z, mc, mf = _scorer()
    ref_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for s in shots:
        p = ref_dir / f"{s['name']}.pkl"
        if not p.exists():
            p.write_bytes(pickle.dumps(ls.build_reference(s["psi"], R, Z, mc, mf)))
        refs.append(pickle.loads(p.read_bytes()))
    return refs


def score(shots, refs, preds) -> dict:
    """Mirror of local_score.main's aggregation, on in-memory shots and predictions."""
    ls, R, Z, mc, mf = _scorer()
    st = {"psi_sum": 0.0, "psi_sumsq": 0.0, "psi_n": 0.0,
          "scal_sum": np.zeros(ls.N_SCALARS), "scal_sumsq": np.zeros(ls.N_SCALARS),
          "scal_n": np.zeros(ls.N_SCALARS), "cons_sum": np.zeros(ls.N_CONS),
          "cons_sumsq": np.zeros(ls.N_CONS), "cons_n": np.zeros(ls.N_CONS)}
    for s, ref in zip(shots, refs):
        g = s["psi"].astype(np.float64)
        st["psi_sum"] += g.sum(); st["psi_sumsq"] += (g ** 2).sum(); st["psi_n"] += g.size
        for j, name in enumerate(ls.SCALARS):
            v = s[name][np.isfinite(s[name])]
            st["scal_sum"][j] += v.sum(); st["scal_sumsq"][j] += (v ** 2).sum(); st["scal_n"][j] += v.size
        for j in range(ls.N_CONS):
            v = ref[1][ref[2][:, j], j]
            st["cons_sum"][j] += v.sum(); st["cons_sumsq"][j] += (v ** 2).sum(); st["cons_n"][j] += v.size
    safe = lambda n: np.where(n > 0, n, 1)
    means = (st["psi_sum"] / st["psi_n"],
             np.where(st["scal_n"] > 0, st["scal_sum"] / safe(st["scal_n"]), 0.0),
             np.where(st["cons_n"] > 0, st["cons_sum"] / safe(st["cons_n"]), 0.0))
    # One global flux sign for the whole set, as on the leaderboard.
    tot = {sg: 0.0 for sg in ls.PSI_SIGNS}
    for s, p in zip(shots, preds):
        rr, _, _ = ls.psi_residuals(s["psi"], p["psirz"], means[0])
        for sg in ls.PSI_SIGNS:
            tot[sg] += rr[sg]
    sign = min(ls.PSI_SIGNS, key=tot.get)
    acc = ls.Accum()
    acc.psi_sign = sign
    for s, ref, p in zip(shots, refs, preds):
        acc.add(ls.score_shot(s, ref, p, R, Z, mc, mf, sign, means))
    return ls.finalize_machine(acc, st)


def _train(a, t0):
    tr = load_packed(a.cache, {int(f) for f in a.train_folds.split(",")}, stride=a.stride)
    print(f"train frames {len(tr['X'])} from {len(np.unique(tr['shot']))} shots ({time.time()-t0:.0f}s)")
    model = REGISTRY[a.model](**a.kw).fit(tr["X"], tr["psi"], tr["y"])
    del tr
    print(f"fit done ({time.time()-t0:.0f}s)")
    if a.save:
        (RUNS_DIR / "models").mkdir(exist_ok=True)
        (RUNS_DIR / "models" / f"{a.tag or a.model}.pkl").write_bytes(pickle.dumps(model))
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pca_ridge", choices=sorted(REGISTRY))
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--stride", type=int, default=4, help="keep every k-th training frame (RAM)")
    ap.add_argument("--train-folds", default="0,1,2,3")
    ap.add_argument("--cache", type=Path, default=RUNS_DIR / "cache")
    ap.add_argument("--kw", type=json.loads, default={}, help='model kwargs, e.g. {"n_components": 128}')
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", action="store_true", help="pickle the fitted model to RUNS_DIR/models")
    ap.add_argument("--ensemble", default="", help="comma-separated saved tags to average; skips training")
    a = ap.parse_args()

    t0 = time.time()
    if a.ensemble:
        model = Ensemble(pickle.loads((RUNS_DIR / "models" / f"{t}.pkl").read_bytes())
                         for t in a.ensemble.split(","))
    else:
        model = _train(a, t0)

    shots = val_shots(a.cache, a.n_val)
    refs = references(shots, RUNS_DIR / "refs")
    preds = []
    for s in shots:
        p = model.predict(s["X"])
        preds.append({"psirz": p.psi.astype(np.float64), "q95": p.scalars[:, 0], "betaN": p.scalars[:, 1]})
    res = score(shots, refs, preds)
    # Pooled R2_psi hides which shots drive the error; list the worst by per-shot residual share.
    ss = [float(((s["psi"] - p["psirz"]) ** 2).sum()) for s, p in zip(shots, preds)]
    worst = sorted(zip(ss, [s["name"] for s in shots], [len(s["psi"]) for s in shots]), reverse=True)[:5]
    print("worst shots (share of SS_res, T):",
          [(n, round(e / sum(ss), 3), t) for e, n, t in worst])
    out = {"model": "ensemble" if a.ensemble else a.model, "kw": a.ensemble or a.kw, "tag": a.tag,
           "n_val": len(shots), "stride": a.stride, "S": res["S"],
           "r2_psi": res["r2_psi"], "r2_qb": res["r2_qb"], "1-dlcfs": 1 - min(1.0, res["dlcfs"]),
           "consistency": res["consistency"], "r2_cons_each": res["r2_cons_each"],
           "secs": round(time.time() - t0)}
    print(json.dumps(out, indent=1, default=float))
    with open(RUNS_DIR / "results.jsonl", "a") as f:
        f.write(json.dumps(out, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
