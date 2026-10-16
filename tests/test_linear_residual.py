import numpy as np

from fusion_eq.models import LinearResidual, Prediction


class _Recorder:
    """Base model that stores the residual it was trained on and predicts zeros."""

    def fit(self, X, psi, scalars):
        self.psi = np.array(psi)
        return self

    def predict(self, X):
        return Prediction(psi=np.zeros((len(X), *self.psi.shape[1:])), scalars=np.zeros((len(X), 2)))


def test_linear_flux_leaves_zero_residual(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(500, 6))  # bounded: the 0.1-99.9 percentile clip moves rows by ~1e-3
    W = rng.normal(size=(6, 4 * 5))
    psi = (X @ W + 0.3).reshape(-1, 4, 5).astype(np.float32)
    m = LinearResidual(_Recorder(), ridge=0.0, chunk=128, scratch=tmp_path).fit(X, psi, np.zeros((500, 2)))
    assert np.abs(m.base.psi).max() < 2e-2 < np.abs(psi).mean()
    Xq = rng.uniform(-0.5, 0.5, size=(7, 6))
    assert np.allclose(m.predict(Xq).psi, (Xq @ W + 0.3).reshape(-1, 4, 5), atol=2e-2)
    assert not list(tmp_path.iterdir())  # residual memmap removed
