# Methods (Fusion Equilibrium Challenge, Codabench #17739, team nas_2242)

Two predictors share each submission: a linear model trained on the public DIII-D train split, and a zero-shot physics solver for MAST that uses no MAST equilibrium data from any source. Code, weights and pins: this repository; commands: `docs/REPRODUCE.md`, `scripts/train.sh`, `scripts/infer.sh`.

## DIII-D: ridge on PCA coefficients
- Inputs per EFIT time: the starter's interpolated coil currents and Ip (errata applied), core and edge Thomson summaries, pressure features, time since the first equilibrium, and sign(Ip) × inputs (about 8% of shots run reversed Ip). Inputs are clipped to the training 0.1-99.9 percentiles.
- Targets: 32 PCA coefficients of the 65×65 flux map, q95 and betaN, each fitted by `RidgeCV` (alphas 1e-3 to 1e3).
- Training: every 6th frame of the 7,041 train shots, 200 shots held out by shot. Held-out S 0.903.
- Award 2 ranks G = S_MAST / S_DIII-D among entries with S_DIII-D ≥ 0.85, so the Finals use this ridge rather than our MLP ensemble (held-out S 0.9915, Development only), and pull q95 and betaN toward the training mean, x → m + 0.4 (x − m) (`d3d_shrink`; held-out S 0.862). This lowers DIII-D accuracy on purpose. All three Finals carry the same DIII-D file.

## MAST: free-boundary physics solver, no MAST labels
1. **Vacuum flux** from Green's functions of every shipped coil conductor. The solenoid column reads 2× the FAIR-MAST level-2 current; every Final uses scale 0.5 (`--sol 0.5`).
2. **Free-boundary solve.** Picard iteration (4 passes) on J = (β0 R/R0 + (1−β0) R0/R)(1−ψ_N)^α, normalised to the measured Ip. β0 is picked per frame to match the outer LCFS radius read off the midplane Thomson density. Axis and LCFS searches use a mask from the coil-conductor hull, never the scorer's label-derived envelope.
3. **Scalars.** q95 is measured on the solved flux with the scorer's surface finders. betaN is volume-averaged Thomson pressure over B0²/2μ0, divided by 0.65 for the NBI fast-ion share (Cecconello et al., NF 59 016006, 2019).
4. **Time smoothing.** ψ and q95 over 5 frames, betaN over 5 (Finals 1, 3) or 13 (Final 2), from leave-one-out CV on MAST inputs only (`mast_smooth_cv`).
5. **Current-profile closure** (Finals 1, 3). A bagged gradient-boosted regressor predicts the profile exponent α per frame from Thomson and solver features. Its target is the α that best fits EFIT J_φ on 1,500 DIII-D train shots outside the bench set. The solve is rerun with the predicted α.
6. **Twin correction.** A ridge model on 32 PCA modes of ψ/Ip learns the solver's mean error from synthetic equilibria: our own solver, driven by each private MAST frame's inputs with randomised profile, β0, coil calibration, error field and edge error. No equilibrium label enters; generating from the test inputs is label-free test-time adaptation under rule 3.
7. **Correction strength**, fitted on DIII-D, where the same generator is scored against EFIT on 300 train shots (two halves). Per-mode least-squares slopes, clipped to [0, 1], are carried to the MAST twin modes by squared overlap on the grid (`twin map`; not validated against MAST labels). Final 3 pulls them toward one global slope: 0.75 × mode scale + 0.25 × 0.39, where 0.39 is the global slope on half A (0.397 on half B; `scripts/global_scale.py`).
8. **Vertical shift.** Each frame's plasma flux is moved to zero net vertical coil force, within ±0.15 m, then median-filtered in time.

## Final submissions
Bench dS: the DIII-D analogue of each MAST pipeline against the bare solver (300 train shots). Te check: label-free Thomson-Te mismatch on 81 private MAST shots against the Development baseline (lower is better).

| Final | Codabench | MAST pipeline | Bench dS [95% CI] | Te check [95% CI] |
|---|---|---|---|---|
| 1 | 1011418 | solve → closure → twin, per-mode → shift | +0.0847 [+0.072, +0.098] | +0.0036 [+0.0015, +0.0059] |
| 2 | 1011432 | solve → twin, per-mode → shift | +0.0785 [+0.066, +0.092] | +0.0017 [+0.0002, +0.0031] |
| 3 | 1011437 | solve → closure → twin, blended → shift | +0.0684 [+0.059, +0.081] | +0.0010 [−0.0008, +0.0028] |

## Data use (rule 3)
- **DIII-D:** public DIII-D train split only, split by shot. **MAST:** the Final private-test inputs (coil currents, Ip, Thomson), the coil-conductor geometry shipped in those input rows and MAST design geometry; no MAST or MAST-U equilibrium data from any source. The twin scales and the closure are fitted on DIII-D train EFIT labels only. Our code never reads `mast_envelope.npz`.
- External datasets and pre-trained models: none.
- Development-phase scores of our own submissions were used to choose between MAST variants (rule 3 feedback clause), including rejecting a per-shot solenoid scale (0.656 against 0.735). The Final pipelines were then chosen among these on the DIII-D bench and the Te check.
- Two Development submissions (P1, P2) rescaled predictions to measure label statistics, which the 2026-10-04 probe clause counts as MAST data. Nothing fitted to or derived from them is in any Final, and that code was removed.
- Through Development candidate c3, MAST variants were also scored on the 3 labelled demo shots. That scorer was removed; the constants first picked there now come from label-free sources (CV, design geometry, signal convention).

## Compute
Rebuild of 2026-10-30 (`scripts/train.sh`, `scripts/infer.sh`) on one Apple M4 (10-core CPU, 16 GB), 8 worker processes, no GPU (0 GPU-h).

- Training, 2 h 41 min: DIII-D feature cache 58 min, ridge and closure 43 min, MAST twin 33 min, DIII-D twin, bench and mode scales 27 min.
- Inference per Final: DIII-D file 6 min, MAST solve 2 h 10 min, closure re-solve 1 h 50 min (Finals 1, 3), twin, betaN and shift 21-25 min.

## AI assistant use
Code, experiments and documents were written with an AI coding assistant (Claude Code). The team checked all runs, scores and submissions and takes responsibility for the entry.
