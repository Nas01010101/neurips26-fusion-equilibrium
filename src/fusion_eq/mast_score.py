"""Score a MAST predictor on the starter's 3 labelled MAST demo shots with the official functions.

These are the only MAST labels anywhere; they are used for evaluation, never fitting. Three shots
from one session is a small, correlated sample: read differences below ~0.02 as noise.

    PYTHONPATH=src uv run python -m fusion_eq.mast_score --kw '{"kappa": 1.9}'
"""
import argparse
import json

import numpy as np
import pandas as pd

from .paths import STARTER, use_starter

DEMOS = ["mast_shot_28348", "mast_shot_28350", "mast_shot_28351"]


def load_demo(name: str) -> dict:
    row = pd.read_parquet(STARTER / "parquet_data" / f"{name}.parquet").iloc[0]
    gt = {"psi": np.stack([np.stack(p) for p in row["efit_psirz"]]).astype(np.float64),
          "q95": np.asarray(row["efit_q95"], float), "betaN": np.asarray(row["efit_beta_n"], float)}
    # The model sees the row with every EFIT-derived column removed, as on the test split.
    inputs = row.drop([c for c in row.index if c.startswith("efit_") and c not in
                       ("efit_times", "efit_grid_R", "efit_grid_Z")] + ["magnetics_dsep",
                                                                        "magnetics_dsep_times"])
    return {"row": inputs, **gt}


def score(predict, shots=None) -> dict:
    use_starter()
    import local_score as ls
    from metrics import Accum, finalize_machine
    ls.MACHINE = "MAST"
    mask = np.load(STARTER / "fusion_scoring" / "masks" / "mast_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mc = mask["mask_coarse"].astype(bool)
    mf = mc.astype(np.float64)
    shots = shots or [load_demo(n) for n in DEMOS]
    refs, stats = [], {k: 0.0 for k in ("psi_sum", "psi_sumsq", "psi_n")}
    stats.update({k: np.zeros(ls.N_SCALARS) for k in ("scal_sum", "scal_sumsq", "scal_n")})
    stats.update({k: np.zeros(ls.N_CONS) for k in ("cons_sum", "cons_sumsq", "cons_n")})
    for s in shots:
        ref = ls.build_reference(s["psi"], R, Z, mc, mf)
        refs.append(ref)
        g = s["psi"]
        stats["psi_sum"] += g.sum(); stats["psi_sumsq"] += (g ** 2).sum(); stats["psi_n"] += g.size
        for j, name in enumerate(ls.SCALARS):
            v = s[name][np.isfinite(s[name])]
            stats["scal_sum"][j] += v.sum(); stats["scal_sumsq"][j] += (v ** 2).sum()
            stats["scal_n"][j] += v.size
        for j in range(ls.N_CONS):
            v = ref[1][ref[2][:, j], j]
            stats["cons_sum"][j] += v.sum(); stats["cons_sumsq"][j] += (v ** 2).sum()
            stats["cons_n"][j] += v.size
    n = lambda a, b: np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)  # noqa: E731
    means = (stats["psi_sum"] / stats["psi_n"], n(stats["scal_sum"], stats["scal_n"]),
             n(stats["cons_sum"], stats["cons_n"]))
    preds = [predict(s["row"]) for s in shots]
    tot = {sg: sum(ls.psi_residuals(s["psi"], p["psirz"], means[0])[0][sg] for s, p in zip(shots, preds))
           for sg in ls.PSI_SIGNS}
    sign = min(ls.PSI_SIGNS, key=tot.get)
    acc = Accum()
    acc.psi_sign = sign
    for s, ref, p in zip(shots, refs, preds):
        acc.add(ls.score_shot(s, ref, p, R, Z, mc, mf, sign, means))
    res = finalize_machine(acc, stats)
    return {"S": res["S"], "r2_psi": res["r2_psi"], "r2_qb": res["r2_qb"],
            "1-dlcfs": 1 - min(1.0, res["dlcfs"]), "consistency": res["consistency"], "r2_q_b": res["r2_each"],
            "cons": res["r2_cons_each"], "lcfs_fail": res["lcfs_fail_frac"], "psi_sign": sign}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kw", default="{}", help="JSON kwargs for predict_mast")
    a = ap.parse_args()
    use_starter()
    from .mast import predict_mast
    kw = json.loads(a.kw)
    res = score(lambda row: predict_mast(row, **kw))
    print(json.dumps({"kw": kw, **res}, indent=1, default=lambda x: round(float(x), 4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
