"""MLP on PCA coefficients of the flux map. Importing this module registers it in REGISTRY.

PCA bases are orthonormal, so squared error on unwhitened coefficients equals pixel MSE inside the
retained subspace: the loss optimizes R2_psi directly instead of treating every mode equally.
"""
import numpy as np
import torch
from torch import nn

from .models import REGISTRY, Prediction, fit_pca, pca_transform, register


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@register("pca_mlp")
class PcaMlp:
    cols = None  # class defaults keep models pickled before these options existed loadable
    clip = 0.0

    def __init__(self, n_components: int = 64, width: int = 512, depth: int = 3,
                 epochs: int = 40, batch: int = 1024, lr: float = 2e-3, seed: int = 0,
                 clip: float = 0.0, cols: list | None = None):
        self.k, self.width, self.depth = n_components, width, depth
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed
        self.cols = cols  # input-column subset, e.g. machine-agnostic Ip+Thomson only for MAST transfer
        self.clip = clip  # grad-norm cap; width 2048 at lr 2e-3 diverged near the OneCycle peak

    def fit(self, X, psi, scalars):
        torch.manual_seed(self.seed)
        self._shape = psi.shape[1:]
        flat = psi.reshape(len(psi), -1)
        self._pca = fit_pca(flat, self.k, seed=self.seed)
        return self.fit_coefs(X, pca_transform(self._pca, flat), scalars)

    def fit_coefs(self, X, c, scalars, pca=None, shape=None):
        """fit() on precomputed PCA coefficients c (N, k): lets every training frame be used
        when the flux maps themselves (N x 65 x 65) would not fit in RAM."""
        if pca is not None:
            self._pca, self._shape = pca, shape
            torch.manual_seed(self.seed)
        X = self._sel(X).astype(np.float64)  # float32 moments overflow on wide-range inputs
        self._lo, self._hi = np.nanpercentile(X, [0.1, 99.9], axis=0)
        X = np.clip(X, self._lo, self._hi)
        self._xm, self._xs = X.mean(0), X.std(0) + 1e-8
        self._cs = float(c.std())  # one global scale keeps the pixel-MSE equivalence
        ok = np.isfinite(scalars)
        self._sm = np.nanmean(scalars, 0)
        self._ss = np.nanstd(scalars, 0) + 1e-8
        s = np.where(ok, (scalars - self._sm) / self._ss, 0.0)

        dev = _device()
        layers, d = [], X.shape[1]
        for _ in range(self.depth):
            layers += [nn.Linear(d, self.width), nn.SiLU()]
            d = self.width
        self._net = nn.Sequential(*layers, nn.Linear(d, self.k + 2)).to(dev)
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)
        Xt, Ct, St, Mt = t((X - self._xm) / self._xs), t(c / self._cs), t(s), t(ok)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * -(-len(Xt) // self.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=steps)
        for ep in range(self.epochs):
            perm = torch.randperm(len(Xt), device=dev)
            tot = 0.0
            for i in range(0, len(Xt), self.batch):
                b = perm[i:i + self.batch]
                out = self._net(Xt[b])
                l_psi = ((out[:, :self.k] - Ct[b]) ** 2).sum(1).mean()
                l_sc = (((out[:, self.k:] - St[b]) ** 2) * Mt[b]).sum() / Mt[b].sum().clamp(min=1)
                loss = l_psi + l_sc
                opt.zero_grad(); loss.backward()
                if self.clip:
                    nn.utils.clip_grad_norm_(self._net.parameters(), self.clip)
                opt.step(); sched.step()
                tot += loss.item() * len(b)
            if ep % 10 == 0 or ep == self.epochs - 1:
                print(f"  epoch {ep} loss {tot / len(Xt):.4f}", flush=True)
        self._net.eval()
        return self

    def _sel(self, X):
        return X if self.cols is None else X[:, self.cols]

    @torch.no_grad()
    def predict(self, X):
        dev = next(self._net.parameters()).device
        x = torch.as_tensor(((np.clip(self._sel(X).astype(np.float64), self._lo, self._hi) - self._xm) / self._xs).astype(np.float32), device=dev)
        out = self._net(x).cpu().numpy().astype(np.float64)
        psi = self._pca.inverse_transform(out[:, :self.k] * self._cs).reshape(len(X), *self._shape)
        return Prediction(psi=psi, scalars=out[:, self.k:] * self._ss + self._sm)


@register("pca_mlp128")
def _pca_mlp128():
    return PcaMlp(n_components=128)


@register("lin_res_mlp")
def _lin_res_mlp(**kw):
    from .models import LinearResidual
    from .paths import RUNS_DIR
    return LinearResidual(PcaMlp(**kw), scratch=RUNS_DIR)  # residual memmap off the internal disk


__all__ = ["PcaMlp", "REGISTRY"]
