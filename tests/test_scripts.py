import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_global_scale_recovers_slope_and_ignores_frame_offsets(tmp_path, monkeypatch):
    gs = _load("global_scale")
    rng = np.random.default_rng(0)
    rows, lines = {}, []
    for st in ("d3d_shot_a", "d3d_shot_b"):
        p0, d = rng.standard_normal((2, 3, 4, 4))
        offset = rng.standard_normal((3, 1, 1))  # per-frame flux offset, removed before the fit
        for sub, psi in (("iter4", p0), ("twin", p0 + d)):
            (tmp_path / sub).mkdir(exist_ok=True)
            np.savez(tmp_path / sub / f"{st}.npz", psirz=psi.astype(np.float32))
        rows[f"{st}.parquet"] = {"efit_psirz": list(p0 + 0.39 * d + offset), "efit_times": [0, 1, 2]}
        lines.append(f"{st}.parquet")
    (tmp_path / "half.txt").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(gs, "load_row", rows.__getitem__)
    monkeypatch.setattr(gs, "frames_of", lambda n: np.arange(n))
    assert abs(gs.global_scale(tmp_path / "half.txt", tmp_path) - 0.39) < 1e-6


def test_npzcmp_reports_key_and_value_differences(tmp_path):
    np.savez(tmp_path / "a.npz", x=np.zeros(3, np.float32), y=np.ones(2), only_a=np.zeros(1))
    np.savez(tmp_path / "b.npz", x=np.array([0, 5e-7, 0], np.float32), y=np.ones(2))
    out = subprocess.run([sys.executable, SCRIPTS / "npzcmp.py", tmp_path / "a.npz", tmp_path / "b.npz"],
                         capture_output=True, text=True, check=True).stdout
    assert "only in A: 1" in out and "only in B: 0" in out and "shared: 2" in out
    assert "arrays differing: 1" in out
    assert abs(float(out.split("max abs diff:")[1]) - 5e-7) < 1e-12
