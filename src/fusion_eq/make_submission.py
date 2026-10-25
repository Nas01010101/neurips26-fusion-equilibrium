"""Run the starter's submission_skeleton with our model plugged into its predict hook.

Reusing the skeleton keeps its shot ordering, validation and push/pointer logic byte-for-byte; only
`your_model_predict` is replaced. DIII-D uses the saved ridge (`DIIID_MODELS`). MAST reads the
per-shot cache that `fusion_eq.mast_batch` writes (the physics predictor is too slow to run
inline); a missing shot is an error, never a silent zero map.

    PYTHONPATH=src uv run python -m fusion_eq.make_submission --max-shots 5     # format check
"""
import os
import pickle
import sys

import numpy as np

from .data import add_thomson, shot_features
from .models import Ensemble
from .paths import RUNS_DIR, SPLIT, use_starter

DIIID_MODELS = ["ridge32-ipx-clip-v200"]
_model = None


class _NumpyPsi32:
    """numpy for the skeleton, with float16 mapped to float32. The skeleton stores psi as float16
    (submission_skeleton.py:102); the scorer upcasts on read, so float32 is accepted, and on the
    validation fold float16 cost 0.0015 consistency (0.0003 S) at the DIII-D level we are at."""

    def __getattr__(self, name):
        return np.float32 if name == "float16" else getattr(np, name)


def row_features(row) -> np.ndarray:
    """Model inputs for one raw dataset row, aligned to its efit_times. Test rows carry no
    efit_psirz, but the starter loader requires the key, so a zero stand-in is supplied."""
    use_starter()
    from experiments import load_shot_from_hf_row
    row = dict(row)
    T = len(np.asarray(row["efit_times"]))
    row["efit_psirz"] = [np.zeros((65, 65))] * T
    X = shot_features(add_thomson(load_shot_from_hf_row(row), row))
    if len(X) < T:
        raise ValueError(f"{len(X)} feature frames < {T} efit_times")
    return X[:T]


def predict_diiid(row, model) -> dict:
    p = model.predict(row_features(row))
    return {"psirz": p.psi.astype(np.float32), "q95": p.scalars[:, 0].astype(np.float32),
            "betaN": p.scalars[:, 1].astype(np.float32)}


def load_mast(i: int, T: int) -> dict:
    from .mast_batch import cache_path
    p = cache_path(i)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run fusion_eq.mast_batch first")
    d = np.load(p)
    if d["psirz"].shape[0] != T:
        raise ValueError(f"MAST shot {i}: cached {d['psirz'].shape[0]} frames != {T} efit_times")
    return {k: d[k] for k in ("psirz", "q95", "betaN")}


def main() -> int:
    global _model
    use_starter()
    import submission_skeleton as sk
    try:  # registers PcaMlp so MLP pickles resolve; the shipped ridge needs no torch
        from . import models_torch  # noqa: F401
    except ImportError:
        pass

    # FUSION_DIIID_MODELS=tag[,tag] swaps the DIII-D model per build; several tags are averaged.
    tags = os.environ.get("FUSION_DIIID_MODELS", ",".join(DIIID_MODELS)).split(",")
    _model = Ensemble(pickle.loads((RUNS_DIR / "models" / f"{t}.pkl").read_bytes()) for t in tags)
    mast_i = iter(range(10 ** 6))  # the skeleton streams MAST rows in the order mast_batch cached

    def predict(row, source):
        if source == "DIII-D":
            return predict_diiid(row, _model)
        return load_mast(next(mast_i), len(np.asarray(row["efit_times"])))

    sk.your_model_predict = predict
    sk.TEST_CONFIGS = [(f"diii_d_{SPLIT}", SPLIT), (f"mast_{SPLIT}", SPLIT)]
    # FUSION_DIIID_ONLY=1 writes the DIII-D file alone while MAST is still solving; swap_mast adds MAST.
    if os.environ.get("FUSION_DIIID_ONLY") == "1":
        sk.TEST_CONFIGS = sk.TEST_CONFIGS[:1]
    sk.np = _NumpyPsi32()
    return sk.main()


if __name__ == "__main__":
    sys.exit(main())
