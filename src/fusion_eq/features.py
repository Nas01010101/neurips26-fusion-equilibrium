"""Machine-agnostic features for the cross-machine challenge.

Raw coil currents are wired differently on DIII-D and MAST, so they cannot transfer. These
features come from quantities both machines measure the same way (Ip, Thomson Te/ne).
"""
import numpy as np

_KEV_J = 1.602e-16  # keV -> J; keeps the pressure proxy in SI-like units


def pressure_features(te: np.ndarray, ne: np.ndarray) -> np.ndarray:
    """Per-timestep summary of the electron pressure profile p = ne * Te.

    te, ne: (T, channels). NaN channels (dropped Thomson chords) are ignored.
    Returns (T, 4): log10 peak, log10 mean, peak/mean (peakedness), edge-to-core ratio.
    Magnitudes are logged and ratios bounded: a near-empty core channel once drove the raw
    edge/core ratio to 1e37, which overflows float32 statistics downstream.
    """
    p = np.asarray(ne, dtype=np.float64) * np.asarray(te, dtype=np.float64) * _KEV_J
    p = np.clip(p, 0.0, None)  # negative pressure is a fit artifact, not physics
    peak = np.nanmax(p, axis=1)
    mean = np.nanmean(p, axis=1)
    floor = 1e-3 * np.maximum(peak, 1e-30)  # ratios relative to this shot's own scale
    peaked = np.clip(peak / np.maximum(mean, floor), 0.0, 100.0)
    n = p.shape[1]
    k = max(n // 4, 1)
    edge = np.clip(np.nanmean(p[:, -k:], axis=1) / np.maximum(np.nanmean(p[:, :k], axis=1), floor),
                   0.0, 100.0)
    return np.column_stack([np.log10(peak + 1.0), np.log10(mean + 1.0), peaked, edge])


def shape_normalize(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample (psi - mean) / range, so models predict shape not machine-specific level.

    Returns (normalized, mean, range); invert with normalized * range + mean.
    """
    flat = psi.reshape(len(psi), -1)
    mean = flat.mean(axis=1)
    rng = np.maximum(np.ptp(flat, axis=1), 1e-12)
    return ((flat - mean[:, None]) / rng[:, None]).reshape(psi.shape), mean, rng
