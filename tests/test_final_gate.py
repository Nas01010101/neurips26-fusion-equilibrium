import numpy as np

from fusion_eq import final_gate as fg


def _bench(gain, sd=0.001, n=300, n_boot=5000):
    rng = np.random.default_rng(0)
    a = 0.55 + sd * rng.standard_normal(n_boot)
    return {"n_shots": n, "S": {"a": 0.55, "c": 0.55 + gain},
            "boot_S": {"a": a.tolist(), "c": (a + gain + sd * rng.standard_normal(n_boot)).tolist()}}


def _case(**kw):
    c = {"compliance": {"ok": True}, "is_anchor": False, "redundant": False, "min_bench_shots": 200,
         "bench": fg.bench_test(_bench(0.02), "c", "a", 0.01, 25),
         "te": {"worse": False}}
    c.update(kw)
    return c


def test_strong_bench_win_uploads():
    assert fg.decide(_case()) == "UPLOAD"


def test_compliance_failure_blocks_even_with_evidence():
    assert fg.decide(_case(compliance={"ok": False})) == "BLOCK"


def test_small_gain_below_bonferroni_bar_holds():
    b = fg.bench_test(_bench(0.003, sd=0.001), "c", "a", 0.01, 25)  # z ~2.1: p ~0.02 > 0.0004
    assert b["p"] > b["p_req"] and fg.decide(_case(bench=b)) == "HOLD"


def test_too_few_draws_cannot_reach_the_bar():
    b = fg.bench_test(_bench(0.02, n_boot=1000), "c", "a", 0.01, 25)  # p floor 1/1001 > 0.0004
    assert b["p"] > b["p_req"] and fg.decide(_case(bench=b)) == "HOLD"


def test_missing_bench_te_veto_redundancy_and_few_shots_hold():
    assert fg.decide(_case(bench=None)) == "HOLD"
    assert fg.decide(_case(te={"worse": True})) == "HOLD"
    assert fg.decide(_case(redundant=True)) == "HOLD"
    assert fg.decide(_case(bench=fg.bench_test(_bench(0.02, n=60), "c", "a", 0.01, 25))) == "HOLD"


def test_anchor_needs_no_bench_win():
    assert fg.decide(_case(is_anchor=True, bench=None, te=None)) == "UPLOAD"


def test_banned_patterns_match_probe_names_not_ordinary_ones():
    pats = ["c4z", "994489", r"\bp[123]\b", "envelope"]
    assert fg.banned_hits({"x": "/runs/preds_c4z/shot"}, pats) == [("x", "c4z")]
    assert fg.banned_hits({"x": "used P1 scores"}, pats) == [("x", r"\bp[123]\b")]
    assert fg.banned_hits({"x": "/final/iter4v_cache kp1 sp1"}, pats) == []


def test_rel_rms_zero_for_identical_and_scale_free():
    a = np.random.default_rng(1).standard_normal((3, 5, 5))
    assert fg.rel_rms(a, a) == 0.0
    assert abs(fg.rel_rms(10 * a + 10 * 0.01, 10 * a) - fg.rel_rms(a + 0.01, a)) < 1e-9
