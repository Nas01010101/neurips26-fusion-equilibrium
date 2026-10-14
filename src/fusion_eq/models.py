"""Model registry. A model maps features (N, F) to a flux map (N, 65, 65) plus q95/betaN."""
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Prediction:
    psi: np.ndarray      # (N, H, W)
    scalars: np.ndarray  # (N, 2): q95, betaN


class Model(Protocol):
    def fit(self, X: np.ndarray, psi: np.ndarray, scalars: np.ndarray) -> "Model": ...
    def predict(self, X: np.ndarray) -> Prediction: ...


IP_COL = 20  # plasma_current in the starter's D3D_MAGNETICS_SIGNALS order (features.shot_features)

REGISTRY: dict[str, Callable[..., Model]] = {}


def register(name: str):
    def deco(factory):
        REGISTRY[name] = factory
        return factory
    return deco


def fit_pca(flat: np.ndarray, k: int, max_fit: int = 40_000, seed: int = 0) -> PCA:
    """PCA on a frame subsample: the leading flux modes are stable well below the full set, and
    sklearn copies the whole matrix, which the 16 GB Mac cannot afford at ~1e5 frames."""
    idx = np.random.default_rng(seed).permutation(len(flat))[:max_fit]
    return PCA(k, svd_solver="randomized", random_state=seed).fit(flat[idx])


def pca_transform(pca: PCA, flat: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    return np.concatenate([pca.transform(flat[i:i + chunk]) for i in range(0, len(flat), chunk)])


@register("pca_ridge")
class PcaRidge:
    """Ridge on PCA coefficients of the flux map; the guide's recommended starting point."""

    def __init__(self, n_components: int = 32, ip_interact: bool = False):
        self.n_components = n_components
        self.ip_interact = ip_interact

    def _x(self, X):
        """Optionally append sign(Ip) * X: ~8% of DIII-D shots run reversed Ip, and the EFIT flux
        does not change linearly across that flip, so one linear map cannot fit both regimes."""
        if not self.ip_interact:
            return X
        return np.hstack([X, np.sign(X[:, IP_COL:IP_COL + 1]) * X])

    def fit(self, X, psi, scalars):
        self._lo, self._hi = np.nanpercentile(X, [0.1, 99.9], axis=0)
        X = self._x(X)
        self._shape = psi.shape[1:]
        flat = psi.reshape(len(psi), -1)
        self._pca = fit_pca(flat, self.n_components)
        self._psi = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        self._psi.fit(X, pca_transform(self._pca, flat))
        self._sc = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        ok = np.isfinite(scalars).all(axis=1)  # EFIT scalars are NaN on incomplete frames
        self._sc.fit(X[ok], scalars[ok])
        return self

    def predict(self, X):
        # Clip to the training range: a linear map extrapolates without bound, and one val shot
        # with out-of-range pressure features alone pushed Consistency from 0.58 to 0.23.
        X = self._x(np.clip(X, self._lo, self._hi))
        psi = self._pca.inverse_transform(self._psi.predict(X)).reshape(len(X), *self._shape)
        return Prediction(psi=psi, scalars=self._sc.predict(X))


@register("pca_ridge64")
def _pca_ridge64():
    return PcaRidge(n_components=64)


class Ensemble:
    """Mean of fitted members' predictions; members are already-trained models (e.g. seeds)."""

    def __init__(self, members):
        self.members = list(members)

    def predict(self, X):
        ps = [m.predict(X) for m in self.members]
        return Prediction(psi=np.mean([p.psi for p in ps], 0), scalars=np.mean([p.scalars for p in ps], 0))
