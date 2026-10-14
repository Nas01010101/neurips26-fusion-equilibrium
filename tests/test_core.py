import numpy as np
import pytest

from fusion_eq.features import pressure_features, shape_normalize
from fusion_eq.models import REGISTRY
from fusion_eq.splits import shot_folds
from fusion_eq.submit import write_npz


def test_folds_never_split_a_shot():
    shots = np.repeat(np.arange(20), 7)
    seen = []
    for tr, va in shot_folds(shots, k=4):
        assert not set(shots[tr]) & set(shots[va])
        seen.extend(np.unique(shots[va]))
    assert sorted(seen) == list(range(20))


def test_pressure_features_ignore_nan_channels():
    te = np.array([[1.0, 2.0, np.nan, 0.5]])
    ne = np.ones((1, 4))
    f = pressure_features(te, ne)
    assert f.shape == (1, 4) and np.isfinite(f).all()


def test_shape_normalize_roundtrip():
    psi = np.random.default_rng(0).normal(size=(3, 5, 5))
    n, m, r = shape_normalize(psi)
    np.testing.assert_allclose(n.reshape(3, -1) * r[:, None] + m[:, None], psi.reshape(3, -1), atol=1e-9)


def test_pca_ridge_recovers_linear_map():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6))
    basis = rng.normal(size=(6, 9 * 9))
    psi = (X @ basis).reshape(200, 9, 9)
    sc = np.column_stack([X[:, 0], X[:, 1] * 2])
    m = REGISTRY["pca_ridge"](n_components=6).fit(X, psi, sc)
    inner = np.abs(X).max(1) < 2  # inputs beyond the training 0.1-99.9 percentiles are clipped
    pred = m.predict(X[inner])
    assert np.abs(pred.psi - psi[inner]).max() < 1e-2
    assert np.abs(pred.scalars - sc[inner]).max() < 1e-2


def test_write_npz_rejects_nan(tmp_path):
    bad = {"0": {"psirz": np.full((2, 65, 65), np.nan), "q95": np.ones(2), "betaN": np.ones(2)}}
    with pytest.raises(ValueError):
        write_npz(tmp_path / "x.npz", bad)


def test_load_cache_stride_keeps_every_kth_frame(tmp_path):
    from fusion_eq.data import load_cache
    for i, fold in enumerate((0, 1)):
        np.savez(tmp_path / f"s{i}.npz", X=np.ones((10, 3)), psi=np.ones((10, 4, 4)),
                 y=np.ones((10, 2)), fold=fold)
    d = load_cache(tmp_path, {0}, stride=3)
    assert len(d["X"]) == 4  # frames 0, 3, 6, 9 of the fold-0 shot only


CACHED = sorted((__import__("fusion_eq.paths").paths.RUNS_DIR / "cache").glob("*.npz"))[:1]


@pytest.mark.skipif(not CACHED, reason="no cached shot on the SSD")
def test_official_scorer_gives_one_for_ground_truth(tmp_path):
    from fusion_eq.evaluate import references, score
    z = np.load(CACHED[0])
    s = {"name": CACHED[0].stem, "psi": z["psi"][:8], "q95": z["y"][:8, 0].astype(float),
         "betaN": z["y"][:8, 1].astype(float)}
    refs = references([s], tmp_path)
    res = score([s], refs, [{"psirz": s["psi"].astype(float), "q95": s["q95"], "betaN": s["betaN"]}])
    assert abs(res["S"] - 1.0) < 1e-4


def test_add_thomson_handles_parquet_object_arrays():
    from fusion_eq.data import add_thomson
    prof = np.empty(3, dtype=object)
    for i in range(3):
        prof[i] = np.full(4, float(i + 1))
    row = {"thomson_core_times": np.arange(3.0), "thomson_core_Te": prof, "thomson_core_ne": prof}
    shot = add_thomson({}, row)
    assert shot["thomson_core"]["Te"].shape == (3, 4) and "thomson_edge" not in shot


def test_pca_ridge_ip_interaction_fits_sign_dependent_map():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 21))
    X[:, 20] = np.where(rng.random(400) < 0.5, -1.0, 1.0) * (1 + rng.random(400))
    psi = (np.abs(X[:, 20:21]) * rng.normal(size=(1, 16))).reshape(400, 4, 4)  # |Ip|: not linear
    sc = np.zeros((400, 2))
    err = lambda m: np.abs(m.fit(X, psi, sc).predict(X).psi - psi).mean()
    from fusion_eq.models import PcaRidge
    assert err(PcaRidge(4, ip_interact=True)) < 0.5 * err(PcaRidge(4))


def test_pressure_features_bounded_when_core_is_empty():
    te = np.array([[0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0]])
    ne = np.full((1, 8), 1e19)
    f = pressure_features(te, ne)
    assert np.isfinite(f).all() and np.abs(f).max() <= 100


def test_pca_ridge_clips_inputs_to_training_range():
    """An input far outside the training range must not extrapolate linearly: one val shot with
    log-pressure features at |z|~11 collapsed Consistency from 0.58 to 0.23."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (500, 3))
    psi = (X @ rng.normal(size=(3, 16))).reshape(500, 4, 4)
    m = REGISTRY["pca_ridge"](n_components=3).fit(X, psi, X[:, :2])
    far, edge = np.array([[1e3, 0.0, 0.0]]), np.array([[X[:, 0].max(), 0.0, 0.0]])
    assert np.allclose(m.predict(far).psi, m.predict(edge).psi, atol=0.05)


def test_ensemble_averages_member_predictions():
    from fusion_eq.models import Ensemble, Prediction

    class Const:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return Prediction(psi=np.full((len(X), 2, 2), self.v), scalars=np.full((len(X), 2), self.v))

    p = Ensemble([Const(1.0), Const(3.0)]).predict(np.zeros((4, 3)))
    assert np.allclose(p.psi, 2.0) and np.allclose(p.scalars, 2.0)


def test_pca_mlp_with_grad_clip_trains_finite():
    pytest.importorskip("torch")
    from fusion_eq.models_torch import PcaMlp

    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 4))
    psi = (X @ rng.normal(size=(4, 16))).reshape(256, 4, 4)
    p = PcaMlp(n_components=4, width=32, depth=2, epochs=3, batch=64, clip=1.0).fit(X, psi, X[:, :2]).predict(X)
    assert np.isfinite(p.psi).all() and np.isfinite(p.scalars).all()


def test_pca_mlp_cols_ignores_unselected_inputs():
    pytest.importorskip("torch")
    from fusion_eq.models_torch import PcaMlp

    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 4))
    psi = (X[:, :2] @ rng.normal(size=(2, 16))).reshape(128, 4, 4)
    m = PcaMlp(n_components=2, width=16, depth=1, epochs=2, batch=64, cols=[0, 1]).fit(X, psi, X[:, :2])
    X2 = X.copy(); X2[:, 2:] = 1e6
    assert np.allclose(m.predict(X).psi, m.predict(X2).psi)


def test_pca_mlp_pickled_before_new_options_still_selects_all_columns():
    pytest.importorskip("torch")
    from fusion_eq.models_torch import PcaMlp

    old = PcaMlp.__new__(PcaMlp)  # instance state lacks cols/clip, as in older pickles
    assert old._sel(np.zeros((2, 5))).shape == (2, 5) and old.clip == 0.0


def test_row_features_match_training_cache():
    """The submission path must build exactly the features the model was trained on."""
    from pathlib import Path
    import pandas as pd
    import pyarrow.parquet as pq
    from fusion_eq.make_submission import row_features
    from fusion_eq.paths import DATA_DIR, RUNS_DIR

    src = DATA_DIR / "data" / "diii_d_train" / "d3d_shot_00000a10ac.parquet"
    npz = RUNS_DIR / "cache" / "d3d_shot_00000a10ac.npz"
    if not (src.exists() and npz.exists()):
        pytest.skip("training data/cache not mounted")
    cols = [c for c in pq.read_schema(src).names if c != "efit_psirz"]  # as on the test split
    X = row_features(pd.read_parquet(src, columns=cols).iloc[0])
    cached = np.load(npz)["X"]
    assert np.allclose(X[:len(cached)], cached, equal_nan=True)
