import importlib

import fusion_eq.paths as paths


def test_fusion_split_defaults_to_public_and_reads_the_env(monkeypatch):
    assert paths.SPLIT == "public_test"
    monkeypatch.setenv("FUSION_SPLIT", "private_test")
    try:
        assert importlib.reload(paths).SPLIT == "private_test"
    finally:
        monkeypatch.delenv("FUSION_SPLIT")
        importlib.reload(paths)
