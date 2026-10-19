"""Upload gate for a Final submission: one verdict from the evidence we are allowed to use.

The Final keeps the best MAST S and the best DIII-D S across our scored Finals, and the DIII-D
file is the same in all of them, so a slot only pays if its MAST file could beat the others'.
A candidate is judged on four things, none of which reads a MAST label or a probe score:

  compliance  DIII-D file is the pinned ridge (sha256); the MAST cache is a whitelisted pipeline
              with no banned name in its path, build log or manifest; the MAST file holds every
              shot and matches the cache on sampled shots.
  bench       its DIII-D bench analogue (d3d_bench --json) beats the anchor's: one-sided
              bootstrap p = (draws with dS <= 0 + 1) / (draws + 1) at or below the Bonferroni level
              for every bench comparison we have scored (alpha / m_tested), on at least
              min_bench_shots shots. The percentile p, not a normal z: per-shot gains are skewed
              (a few shots carry most of them), which inflates the bootstrap sd.
  te veto     Thomson Te monotonicity on the private MAST shots (te_check_mast) is not
              significantly worse than the anchor's (95% paired-bootstrap CI).
  redundancy  its psi differs from every other slot's by more than redundant_rel_rms.

Verdicts: BLOCK (compliance fails, never upload), HOLD (evidence missing or not strong enough),
UPLOAD. The anchor slot (the default pipeline) needs no bench win. A separate projection
adds the bench gain to the anchor's Development score and compares G with the leaderboard bar;
it assumes the DIII-D gain transfers 1:1 to MAST, which is unverified, so it informs, it does
not gate.

    PYTHONPATH=src uv run python -m fusion_eq.final_gate SUB --cache CACHE --rows ROWS \
        --bench bench300_finals.json [--others CACHE ...] [--json OUT]
"""
import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parents[2] / "docs" / "final_gate.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def banned_hits(texts: dict, patterns) -> list:
    """(source, pattern) for every banned pattern found, case-insensitive."""
    return [(src, p) for src, t in texts.items() for p in patterns if re.search(p, t, re.I)]


def bench_test(bench: dict, cand: str, anchor: str, alpha: float, m: int) -> dict:
    """Paired bootstrap of S(cand) - S(anchor) on the shared draws; one-sided p against alpha / m."""
    d = np.asarray(bench["boot_S"][cand]) - np.asarray(bench["boot_S"][anchor])
    ds = bench["S"][cand] - bench["S"][anchor]
    se = float(d.std(ddof=1)) or 1e-12
    return {"dS": ds, "se": se, "z": ds / se, "p": float(((d <= 0).sum() + 1) / (len(d) + 1)),
            "p_req": alpha / m, "n_boot": len(d),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "n_shots": bench["n_shots"]}


def paired_ci(diff, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def rel_rms(a, b) -> float:
    """||a - b|| / ||b - mean(b)|| over all frames of one shot."""
    b = b.astype(np.float64)
    return float(np.linalg.norm(a.astype(np.float64) - b) / max(np.linalg.norm(b - b.mean()), 1e-30))


def decide(c: dict) -> str:
    if not c["compliance"]["ok"]:
        return "BLOCK"
    if c["is_anchor"]:
        return "UPLOAD" if not c["redundant"] else "HOLD"
    b, te = c.get("bench"), c.get("te")
    if b is None or te is None or c["redundant"]:
        return "HOLD"
    if b["n_shots"] < c["min_bench_shots"] or b["p"] > b["p_req"] or te["worse"]:
        return "HOLD"
    return "UPLOAD"


def check(sub: Path, cache: Path, rows: Path, bench_path: Path | None, others, cfg: dict) -> dict:
    split = "private_test"
    name = cache.name
    texts = {"cache path": str(cache.resolve())}
    log = cache.parent / (name.removesuffix("_cache") + ".log")
    if log.exists():
        texts[log.name] = log.read_text(errors="replace")
    if (sub / "manifest.json").exists():  # repo id and revision; the token is never printed
        texts["manifest"] = re.sub(r"hf_[A-Za-z0-9]+", "", (sub / "manifest.json").read_text())
    comp = {"diiid_sha_ok": sha256(sub / f"diii_d_{split}.npz") == cfg["diiid_sha256"],
            "whitelisted": name in cfg["analogues"],
            "banned": banned_hits(texts, cfg["banned"])}
    m = np.load(sub / f"mast_{split}.npz")
    shots = sorted({k[5:9] for k in m.files})
    comp["n_shots_ok"] = len(shots) == cfg["n_mast"]
    rng = np.random.default_rng(0)
    sample = sorted(rng.choice(cfg["n_mast"], 20, replace=False))
    comp["matches_cache"] = all(
        np.array_equal(m[f"shot_{i:04d}_psirz"], np.load(cache / f"shot_{i:04d}.npz")["psirz"].astype(np.float16))
        for i in sample)
    comp["ok"] = (comp["diiid_sha_ok"] and comp["whitelisted"] and not comp["banned"]
                  and comp["n_shots_ok"] and comp["matches_cache"])
    anchor = cfg["anchor"]
    out = {"sub": str(sub), "cache": name, "compliance": comp, "is_anchor": name == anchor["cache"],
           "min_bench_shots": cfg["min_bench_shots"]}

    # redundancy against every other slot, on a fixed shot sample
    red = {}
    for o in others:
        red[o.name] = float(np.median([rel_rms(np.load(cache / f"shot_{i:04d}.npz")["psirz"],
                                                np.load(o / f"shot_{i:04d}.npz")["psirz"]) for i in sample]))
    out["rel_rms_vs"] = red
    out["redundant"] = any(v < cfg["redundant_rel_rms"] for v in red.values())
    if out["is_anchor"]:
        out["projection"] = projection(anchor, 0.0, cfg)
        return out

    analogue = cfg["analogues"].get(name)
    if bench_path and analogue:
        bench = json.loads(Path(bench_path).read_text())
        if analogue in bench["S"] and anchor["analogue"] in bench["S"]:
            out["bench"] = bench_test(bench, analogue, anchor["analogue"], cfg["alpha"], cfg["m_tested"])
            out["bench"]["analogue"] = analogue

    from .te_check_mast import shot_errs  # loads the starter kit; only needed past this point
    acache = cache.parent / anchor["cache"]
    errs = []
    for i in range(0, cfg["n_mast"], cfg["te_every_shot"]):
        row = pickle.load(open(rows / f"{i:04d}.pkl", "rb"))
        e = shot_errs(row, {"c": np.load(cache / f"shot_{i:04d}.npz")["psirz"],
                            "a": np.load(acache / f"shot_{i:04d}.npz")["psirz"]})
        errs.append((e["c"], e["a"]))
    E = np.array(errs)
    E = E[np.isfinite(E).all(1)]
    d = E[:, 0] - E[:, 1]
    lo, hi = paired_ci(d)
    out["te"] = {"n_shots": len(d), "d_err": float(d.mean()), "ci95": [lo, hi], "worse": lo > 0}

    if "bench" in out:
        out["projection"] = projection(anchor, out["bench"]["dS"], cfg)
    return out


def projection(anchor: dict, ds: float, cfg: dict) -> dict:
    g = (anchor["mast_S"] + ds) / anchor["diiid_S"]
    return {"G": g, "bar": cfg["bar"]["G"], "gap": g - cfg["bar"]["G"],
            "assumes": "bench dS transfers 1:1 to MAST (unverified)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sub", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--bench", type=Path)
    ap.add_argument("--others", type=Path, nargs="*", default=[])
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    r = check(a.sub, a.cache, a.rows, a.bench, a.others, cfg)
    r["verdict"] = decide(r)
    c = r["compliance"]
    print(f"{r['cache']}: {r['verdict']}")
    print(f"  compliance ok={c['ok']} diiid_sha={c['diiid_sha_ok']} whitelisted={c['whitelisted']} "
          f"shots={c['n_shots_ok']} matches_cache={c['matches_cache']} banned={c['banned']}")
    print("  rel_rms vs others " + " ".join(f"{k} {v:.4f}" for k, v in r["rel_rms_vs"].items())
          + ("  REDUNDANT" if r["redundant"] else ""))
    if "bench" in r:
        b = r["bench"]
        print(f"  bench {b['analogue']} vs anchor: dS {b['dS']:+.4f} CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] "
              f"p {b['p']:.5f} (need <= {b['p_req']:.5f}, {b['n_boot']} draws) z {b['z']:.1f} n {b['n_shots']}")
    elif not r["is_anchor"]:
        print("  bench: no analogue scored -> HOLD")
    if "te" in r:
        t = r["te"]
        print(f"  te_check vs anchor: {t['d_err']:+.4f} CI [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] "
              f"n {t['n_shots']}" + ("  WORSE" if t["worse"] else ""))
    if "projection" in r:
        p = r["projection"]
        print(f"  projection G {p['G']:.3f} vs bar {p['bar']:.2f} (gap {p['gap']:+.3f}; {p['assumes']})")
    if a.json:
        a.json.write_text(json.dumps(r, indent=1, default=str))
    return 0 if r["verdict"] == "UPLOAD" else 1


if __name__ == "__main__":
    raise SystemExit(main())
