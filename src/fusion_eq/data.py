"""Per-shot feature cache built from the local parquet download.

Parsing a shot (nested lists, 65x65xT flux maps) dominates load time, so each shot is parsed once
into a compact npz: X (T, F) inputs on the efit_times base, psi (T, 65, 65), y (T, 2) = q95, betaN.
Folds are assigned from a hash of the file name so they are stable across machines and runs.
"""
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .features import pressure_features
from .paths import use_starter

N_FOLDS = 5


def fold_of(name: str, k: int = N_FOLDS) -> int:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % k


def _thomson_at(times_out, system: dict) -> np.ndarray:
    """Interpolate each Thomson channel to efit_times (NaN outside its window), then summarize."""
    t = np.asarray(system["times"], dtype=np.float64)
    out = []
    for key in ("Te", "ne"):
        prof = np.asarray(system[key], dtype=np.float64)
        cols = np.full((len(times_out), prof.shape[1]), np.nan)
        for c in range(prof.shape[1]):
            ok = np.isfinite(prof[:, c])
            if ok.sum() >= 2:
                cols[:, c] = np.interp(times_out, t[ok], prof[ok, c], left=np.nan, right=np.nan)
        out.append(cols)
    with np.errstate(all="ignore"):
        feats = pressure_features(*out)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def add_thomson(shot: dict, row) -> dict:
    """Attach Thomson profiles the starter loader silently drops.

    Parquet yields each profile as an object array of per-time channel arrays; the starter's
    np.asarray(..., float32) raises on that and its bare `except: pass` discards the system.
    """
    for sys_ in ("core", "edge"):
        k = f"thomson_{sys_}"
        if f"{k}_times" not in row or len(row[f"{k}_Te"]) == 0:
            continue
        shot[k] = {"times": np.asarray(row[f"{k}_times"], dtype=np.float64),
                   "Te": np.stack(row[f"{k}_Te"]).astype(np.float64),
                   "ne": np.stack(row[f"{k}_ne"]).astype(np.float64)}
    return shot


def shot_features(shot: dict) -> np.ndarray:
    """Coil currents + Ip (starter interpolation, Ip errata already fixed) + Thomson summaries."""
    from experiments import interpolate_magnetics_to_efit
    t = shot["efit_times"]
    parts = [interpolate_magnetics_to_efit(shot).astype(np.float64)]
    for key in ("thomson_core", "thomson_edge"):
        parts.append(_thomson_at(t, shot[key]) if key in shot else np.zeros((len(t), 4)))
    # Time since first equilibrium: ramp-up vs flat-top shapes differ at equal coil currents.
    parts.append((t - t[0])[:, None])
    return np.hstack(parts).astype(np.float32)


def cache_shot(src: Path, dst_dir: Path) -> str:
    use_starter()
    from experiments import load_shot_from_hf_row
    dst = dst_dir / (src.stem + ".npz")
    if dst.exists():
        return dst.name
    row = pd.read_parquet(src).iloc[0]
    shot = add_thomson(load_shot_from_hf_row(row), row)
    X = shot_features(shot)
    psi = shot["efit_psirz"].astype(np.float32)
    sc = shot.get("scalars", {})
    T = min(len(X), len(psi))
    y = np.full((T, 2), np.nan, dtype=np.float32)
    for j, name in enumerate(("efit_q95", "efit_beta_n")):
        if name in sc:
            m = min(T, len(sc[name]))
            y[:m, j] = sc[name][:m]
    np.savez(dst, X=X[:T], psi=psi[:T], y=y, fold=fold_of(src.name))
    return dst.name


def refresh_features(npz: Path, src_dir: Path) -> str:
    """Recompute X for a cached shot, reusing its psi/y. Skips the efit_psirz column, which is
    most of each parquet file, so a feature change costs minutes instead of a full rebuild."""
    use_starter()
    import pyarrow.parquet as pq
    from experiments import load_shot_from_hf_row
    src = Path(src_dir) / (npz.stem + ".parquet")
    cols = [c for c in pq.read_schema(src).names if c != "efit_psirz"]
    row = pd.read_parquet(src, columns=cols).iloc[0].copy()
    z = dict(np.load(npz))
    T = len(z["psi"])
    row["efit_psirz"] = [np.zeros((65, 65))] * len(row["efit_times"])  # loader requires the key
    X = shot_features(add_thomson(load_shot_from_hf_row(row), row))
    if len(X) < T:
        raise ValueError(f"{npz.name}: {len(X)} feature frames < {T} cached frames")
    z["X"] = X[:T]
    np.savez(npz, **z)
    return npz.name


def build_cache(src_dir: Path, dst_dir: Path, workers: int = 8) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(src_dir).glob("*.parquet"))
    with ProcessPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(cache_shot, files, [dst_dir] * len(files), chunksize=8)):
            if i % 500 == 0:
                print(f"  cached {i}/{len(files)}", flush=True)
    return len(files)


def load_packed(cache_dir: Path, folds: set[int], stride: int = 1) -> dict:
    """load_cache, memoized as one .npy per array: thousands of small npz opens cost ~10 min per
    run on the USB SSD, one sequential read costs seconds. Delete the pack after a feature change."""
    pack = Path(cache_dir).parent / f"pack-f{''.join(map(str, sorted(folds)))}-s{stride}"
    keys = ("X", "psi", "y", "shot")
    if not all((pack / f"{k}.npy").exists() for k in keys):
        d = load_cache(cache_dir, folds, stride)
        pack.mkdir(exist_ok=True)
        for k in keys:
            np.save(pack / f"{k}.npy", d[k])
        return d
    return {k: np.load(pack / f"{k}.npy") for k in keys}


def load_cache(cache_dir: Path, folds: set[int], stride: int = 1) -> dict:
    """Concatenate cached shots in `folds`; drops frames with non-finite inputs or flux maps.

    stride > 1 keeps every k-th frame: neighbouring equilibria are near-duplicates, and the full
    DIII-D set (~1M frames of 65x65 float32) does not fit in 16 GB of RAM.
    """
    X, psi, y, sid = [], [], [], []
    for i, f in enumerate(sorted(Path(cache_dir).glob("*.npz"))):
        if fold_of(f.stem + ".parquet") not in folds:  # skip without opening the file
            continue
        z = np.load(f)
        ok = np.isfinite(z["X"]).all(1) & np.isfinite(z["psi"].reshape(len(z["psi"]), -1)).all(1)
        ok &= np.arange(len(ok)) % stride == 0
        X.append(z["X"][ok]); psi.append(z["psi"][ok]); y.append(z["y"][ok])
        sid.append(np.full(ok.sum(), i, dtype=np.int32))
    return {"X": np.concatenate(X), "psi": np.concatenate(psi), "y": np.concatenate(y),
            "shot": np.concatenate(sid)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refresh", action="store_true", help="recompute X in an existing cache")
    a = ap.parse_args()
    if a.refresh:
        files = sorted(a.dst.glob("*.npz"))
        with ProcessPoolExecutor(a.workers) as ex:
            for i, _ in enumerate(ex.map(refresh_features, files, [a.src] * len(files), chunksize=8)):
                if i % 500 == 0:
                    print(f"  refreshed {i}/{len(files)}", flush=True)
        raise SystemExit(print(f"refreshed {len(files)} shots"))
    print(f"cached {build_cache(a.src, a.dst, a.workers)} shots -> {a.dst}")
