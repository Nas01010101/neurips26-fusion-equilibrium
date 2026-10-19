---
name: final-gate
description: |
  Upload gate for Fusion Equilibrium Final submissions (Codabench #17739, Award 2, G = S_MAST / S_DIII-D).
  Decides BLOCK / HOLD / UPLOAD for a built Final from rule-3-clean evidence only: pinned ridge DIII-D
  sha256, whitelisted MAST pipeline, DIII-D bench analogue win at a Bonferroni-corrected z, Thomson Te
  veto on the private MAST shots, and non-redundancy with the other slots, plus a projected G against
  the leaderboard bar. Fire on any "should I upload", "is this one safe / likely to win", "which Final",
  "swap slot N", or before handing the user a pointer zip.
---

# Final gate

Scoring facts the gate is built on (docs/RULES.md, STATE.md):
- S = 0.55 R²ψ + 0.15 R² q95/βN + 0.10 (1 − D_LCFS) + 0.20 Consistency, with Consistency = mean over
  7 scalars of max(0, R²). The c4 loss budget is consistency 0.145, q95/βN 0.089, D 0.019, ψ 0.012.
  Shape scalars and q95 decide ranks; ψ is saturated.
- The best MAST S and the best DIII-D S across scored Finals both count. The DIII-D file is the same
  in every Final, so G is set by the best MAST file. A slot pays only if its MAST file could be the best.
- MAST labels, probe submissions (P1 994489, P2 994472) and constants fitted to them are banned from
  any Final and from its justification. Dev scores of genuine candidates may select pipelines.

## Procedure
1. **Bench analogue.** Every Final MAST cache maps to a `d3d_bench` variant (`docs/final_gate.json`
   `analogues`). A new pipeline needs a new variant that applies the same operation, not a cousin:
   the in-loop `vert` is not the post-hoc `mast_vertical` shift, which is `pv` / `iter4pv`. Run it on
   rtx (anas@100.88.179.78, `~/fusion-eq`) on `bench300_files.txt`, then score every analogue in one
   call so they share bootstrap draws:
   `d3d_bench --files ../bench300_files.txt --score sym iter4 ... --out ../benchA --json ../bench300_finals.json`.
   Copy the JSON back. Raise `m_tested` in the config by one for every new variant scored, including
   the losers. That is the multiple-comparison count.
2. **Gate each built Final:**
   `PYTHONPATH=src uv run python -m fusion_eq.final_gate final/subN --cache final/X_cache --rows final/rows_private --bench bench300_finals.json --others final/<other slot caches> --json final/subN/gate.json`
   Exit code 0 means UPLOAD.
3. **Read the verdict:**
   - **BLOCK**: compliance failed. Never upload; fix the build.
   - **HOLD**: the evidence is missing or too weak. Typical causes are a bench z below the bar, n < 200, a Te-worse veto, or a near-duplicate of another slot. Do not upload. Either get the evidence or leave the slot to a candidate that has it.
   - **UPLOAD**: the candidate beats the anchor with high confidence and adds something the other slots don't.
4. **Report the projection honestly.** G_proj = (anchor Dev MAST S + bench dS) / ridge Dev DIII-D S, against the Dev
   top-G bar. It assumes 1:1 transfer from DIII-D to MAST, which is unverified. MAST te_check flags iter4 worse (+0.0039, ~70% of c7sv's gap; it caught c7sv, missed c6 on Dev). Older
   Thomson proxies ranked Dev submissions backwards (docs/BENCH.md). So it is information, not a gate. If
   the gap to the bar is larger than any bench dS, say plainly that no candidate makes a win likely.
   Then say which slots still raise our best-of score.
5. **Hand off.** Give the user the verdict line per slot and the pointer zip paths. The user does all
   Codabench uploads. Write the verdicts and the gate JSON paths into STATE.md.

## Rules
- Never relax a threshold to turn HOLD into UPLOAD after seeing the number. The settings are alpha 0.01, Bonferroni over
  m_tested, the 95% Te CI, and redundancy 1e-3. Change them only before a run, with the reason written in STATE.md.
- Only UPLOAD is "very likely better than the anchor". A HOLD candidate added next to the anchor
  cannot lower the score (best MAST across Finals counts), so it is an option, not a risk. Report it
  as "HOLD, harmless under best-of". The user decides whether to fill a slot with one. Never call it likely to win.
- The gate checks provenance by name and log text. It cannot detect a probe-fitted constant baked
  into code, so any new constant needs a cited source in docs/REPRODUCE.md before its cache is whitelisted.
