"""Shot-level splits. Timesteps inside a shot are highly correlated, so rows never straddle folds."""
import numpy as np


def shot_folds(shot_ids: np.ndarray, k: int = 5, seed: int = 0):
    """Yield (train_idx, val_idx) row indices; every shot lands in exactly one val fold."""
    shot_ids = np.asarray(shot_ids)
    shots = np.unique(shot_ids)
    if k > len(shots):
        raise ValueError(f"{k} folds requested for {len(shots)} shots")
    rng = np.random.default_rng(seed)
    rng.shuffle(shots)
    for part in np.array_split(shots, k):
        val = np.isin(shot_ids, part)
        yield np.flatnonzero(~val), np.flatnonzero(val)
