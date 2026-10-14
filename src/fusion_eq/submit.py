"""Write predictions in the Codabench contract: one .npz per config, keys shot_XXXX_{psirz,q95,betaN}.

Arrays are saved at native precision. Rounding or dropping frames destroys the Consistency term
(the starter notes np.round(psi, 3) costs ~35% of it on MAST).
"""
from pathlib import Path

import numpy as np


def write_npz(out: Path, shots: dict[str, dict[str, np.ndarray]]) -> Path:
    """shots: {"0000": {"psirz": (T,65,65), "q95": (T,), "betaN": (T,)}}"""
    arrays = {}
    for sid, d in shots.items():
        T = len(d["q95"])
        if d["psirz"].shape != (T, 65, 65) or len(d["betaN"]) != T:
            raise ValueError(f"shot {sid}: inconsistent shapes")
        if not all(np.isfinite(d[k]).all() for k in ("psirz", "q95", "betaN")):
            raise ValueError(f"shot {sid}: non-finite values")
        for k in ("psirz", "q95", "betaN"):
            arrays[f"shot_{sid}_{k}"] = d[k]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return out
