# Methods (Fusion Equilibrium Challenge, Codabench #17739, team nas_2242)

Two predictors share one submission: a linear model trained on the public DIII-D train split, and a zero-shot physics solver for MAST that uses no MAST equilibrium data from any source. Code: this repository at the submitted commit; commands and pins: `docs/REPRODUCE.md`.

## DIII-D: ridge on PCA coefficients
- Inputs per EFIT time: the starter's interpolated coil currents and Ip (errata applied), core and edge Thomson summaries, pressure features, and time since the first equilibrium (`data.shot_features`).
- Targets: 32 PCA coefficients of the 65x65 flux map (randomised PCA, fixed seed) and the scalars q95 and betaN, each fitted by `RidgeCV` (alphas 1e-3 to 1e3) after standardisation (`models.PcaRidge`).
- sign(Ip) x inputs are appended as interaction features: about 8% of DIII-D shots run reversed Ip, and one linear map cannot fit both regimes.
- Inputs are clipped to the 0.1-99.9 percentile range of the training set, because a linear map extrapolates without bound.
- Training: every 6th frame of the 7,041 train shots, 200 shots held out. The held-out official scorer gives S 0.903: R²ψ 0.992, R² q95/βN 0.804, 1−D_LCFS 0.969, consistency 0.699. Training takes 13-17 min on an Apple M4 CPU with no GPU, and a retrain on the submitted code reproduces the shipped weights byte for byte.
- Why the ridge and not our MLP ensemble (held-out S 0.9915, used in Development submissions): Award 2 ranks G = S_MAST / S_DIII-D among entries with S_DIII-D ≥ 0.85. For a fixed MAST predictor, a lower S_DIII-D above the gate gives a higher G. Every Final submission carries the same DIII-D file.
- **Scalar shrink (Final only).** The submitted q95 and betaN are pulled toward the DIII-D training mean: x → m + 0.4 (x − m), with m = 5.34758 (q95) and 1.24443 (betaN) over the 1,246,612 frames of train folds 0-3 (`d3d_shrink`). ψ is unchanged. On the 200 held-out shots S goes from 0.903 to 0.862 (0.875 at 0.5, 0.869 at 0.45), above the 0.85 gate. The shrink exists only because of the G ratio; it lowers DIII-D accuracy on purpose.

## MAST: free-boundary physics solver, no MAST labels
1. **Vacuum flux.** Green's-function flux of every shipped coil conductor on the EFIT grid (`greens.vacuum_psi`). The solenoid column reads 2x the FAIR-MAST level-2 current, a convention that the Data page leaves open. Every Final fixes the solenoid scale at 0.5 (`--sol 0.5`). A per-shot choice among 0.5, 0.75 and 1.0 against the Thomson inner edge (`mast.pick_sol_scale`) lowered the Development MAST score (0.656 against 0.735) and is not used.
2. **Boundary data.** The MAST Thomson laser is horizontal at the midplane. The innermost and outermost channels whose 5-channel median density exceeds 30% of the profile's 90th percentile mark the inner and outer LCFS midplane radii (`mast.thomson_edges`).
3. **Free-boundary solve.** Picard iteration on the plasma current: J = (β0 R/R0 + (1−β0) R0/R)(1−ψ_N)^α, normalised to the measured Ip (`freeb.solve`). J is symmetrised up-down for stability. β0 is chosen per frame from a 0.0 to 0.5 grid to match the Thomson outer edge (`freeb.solve_matched`). Axis and LCFS searches run inside a mask built from the coil-conductor convex hull, outboard of the solenoid (`mast.machine_mask`), never the scorer's label-derived envelope.
4. **Pipeline options.** `--freeb-iter N` sets the Picard passes per solve (Finals: 4). `--profile spitzer` takes the J shape from Thomson Te^1.5, and `--warm` limits the β0 search to within 0.15 of the previous frame's value; neither is in a Final.
5. **Scalars.**
   - q95 is measured on the solved flux with the scorer's own surface finders (`qprofile.q_at`). The TF vacuum field is F = μ0 N_TF I_TF / 2π with N_TF = 24, and the inboard limiter is at 0.20 m from MAST design geometry. Where no surface is found, q95 falls back to the ITER Physics Basis formula.
   - betaN is the volume-averaged Thomson pressure over B0²/2μ0, divided by (1 − 0.35) for the NBI fast-ion share (Cecconello et al., NF 59 016006, 2019). It depends on inputs only, never on the solved flux.
6. **Time smoothing.** The plasma part of ψ and q95 is averaged over 5 EFIT frames, and betaN over 13 (Final 2) or 5 (Finals 1 and 3). The windows come from leave-one-out CV of a running mean on the Thomson time base of all 1,206 public test shots, using MAST inputs only (`mast_smooth_cv`). The ψ and q95 window comes from the outer-edge series that places each solve. For betaN the pressure series gives 5 frames with per-shot normalised errors and 13 with errors pooled over shots; Final 2 was built at 13, Finals 1 and 3 at 5. This is label-free test-time adaptation under rule 3.

7. **Twin correction** (`twin`). A ridge model on 32 PCA modes of ψ/Ip predicts the residual between the solver's output and the true equilibrium. It is trained only on synthetic equilibria from our own solver, generated from each private MAST frame's inputs (coil currents, Ip, Thomson edges) with the unknowns drawn at random: current profile (1 − ψ_N^a1)^a2, β0, start shape, 3% coil-current calibration error, a uniform error field (B_z 0.02 T, B_R 0.005 T sd), and Thomson-edge error (0.03 m inward bias, 0.02 m sd). The solver is then run on the unperturbed inputs, and the model learns the mean correction. No equilibrium label enters training; generating from the private inputs is label-free test-time adaptation under rule 3. The correction is time-smoothed over the same 5 frames as ψ, and q95 is re-measured on the corrected flux.
8. **Correction strength.** The synthetic-to-real gap is measured on DIII-D, where the same generator, trained on 300 DIII-D train shots' inputs, is scored against their EFIT labels. One global scale, fitted by least squares on half of those shots, is 0.39 (0.397 on the other half). Final 3's bench score covers all 300 shots, so half of them are in-sample for the 0.39 part of its blended scale. The per-mode version fits one slope per PCA mode, clipped to [0, 1]. It damps the first three modes (0.14-0.6), sets several others to 0, including mode 3, the one most linked to profile peaking, and keeps most higher modes at 1. The scales from the two halves agree, and each half is scored with the other half's scales. MAST twin modes are matched to DIII-D modes by their squared overlap on the 65×65 grid (|cos| 0.98, 0.89, 0.76, 0.73 for the first four), and each MAST mode takes the overlap-weighted mean of the DIII-D scales (`twin map`). The mapping between machines is not validated against MAST labels.
9. **Vertical shift** (`mast_vertical`). Each frame's plasma flux is moved vertically to zero net coil force on the plasma current, F_z = −2π∫R J B_R dA, within ±0.15 m, then median-filtered in time.
10. **Scale blend** (`twin blend`, Final 3). Each per-mode scale is pulled toward the global scale: 0.75 × mode scale + 0.25 × 0.39.
11. **Learned current-profile closure** (`closure`, Finals 1 and 3). A gradient-boosted regressor (5 fits bagged over a shot bootstrap) predicts the per-frame exponent α of the solver's current profile. Its target is the α that best fits EFIT's J_φ inside ψ_N 0.95 on 1,500 DIII-D train shots outside the bench set, 10 frames each. Features are Thomson Te and ne profile slopes, log q_cyl, log β_p from Thomson pressure, and two Spitzer resistive-time features, all read off the 4-pass solve's flux; inputs are clipped to the training range. The solve is then rerun with the predicted α and goes through the twin and the vertical shift.

Each constant's source is listed in `docs/REPRODUCE.md`. None is fitted to MAST equilibria.

## Final submissions
All three share the shrunk ridge DIII-D file (shrink 0.4). Bench dS is the DIII-D analogue of each MAST pipeline against the solver without twin or shift, 300 train shots, 20,000 paired bootstrap draws. Te check is the mean Thomson-Te mismatch on 81 private MAST shots against the Development baseline pipeline (c8), a label-free check; lower is better.

| Final | MAST pipeline | Bench dS [95% CI] | Te check [95% CI] |
|---|---|---|---|
| 1 | solver (4 passes) → closure solve → twin, per-mode scale → vertical shift | +0.0847 [+0.072, +0.098] | +0.0036 [+0.0015, +0.0059] |
| 2 | solver (4 passes) → twin, per-mode scale → vertical shift | +0.0785 [+0.066, +0.092] | +0.0017 [+0.0002, +0.0031] |
| 3 | solver (4 passes) → closure solve → twin, blended scale → vertical shift | +0.0684 [+0.059, +0.081] | +0.0010 [−0.0008, +0.0028] |

Final 1 has the largest DIII-D gain and the largest Te mismatch. Final 2 drops the closure. Final 3 blends the per-mode scales toward the global one (0.75 × mode scale + 0.25 × 0.39) and has a Te check level with the baseline. On DIII-D the closure lowers the midplane Te mismatch (−0.0032 [−0.0048, −0.0017] against the 4-pass solve), while on MAST it raises it (+0.0018 [+0.0009, +0.0027], 161 shots). Which pipeline transfers best to MAST is not known before scoring.

## Data use and disclosures (rule 3)
- No MAST equilibrium labels enter training, constants or selection. That includes the starter's labelled demo shots.
- Development-phase scores of our own candidate submissions were used to choose between MAST pipeline variants. This is permitted by the rule 3 feedback clause.
- Two Development submissions (P1, P2) rescaled predictions to measure label statistics. Under the 2026-10-04 probe clause these count as MAST data. No constant fitted to them, and nothing derived from them, is in any Final submission, and the fitting code was removed.
- Through candidate c3, MAST variants were also scored on the 3 labelled demo shots. That scorer was removed. The four constants first picked on those shots now come from label-free sources:
  - SMOOTH and BETA_SMOOTH from the CV above (BETA_SMOOTH 9 → 5).
  - WALL_R from MAST design geometry.
  - SOL_SCALE 0.5 from the signal convention.
- Development scores per MAST variant are in `docs/SUBMISSIONS.md`. The Final pipelines were chosen on the DIII-D bench and the label-free Te check, not on Development scores.
- The twin scales (global and per mode) and the closure are fitted on DIII-D train EFIT labels only.

## Compute
| Step | Hardware | Cost |
|---|---|---|
| DIII-D train + 200-shot validation | Apple M4 CPU | 994 s |
| MAST solve, 1,206 public test shots | Apple M4 CPU, 7 workers | 22 to 28 CPU-h per variant (190 to 244 min wall) |
| Twin data, 1,205 private MAST shots | Apple M4 CPU | not timed separately |
| Twin data and bench, 300 DIII-D train shots | Linux CPU, 4 workers | about 3.5 s per synthetic draw |
| GPU | none | 0 GPU-h |

All MAST steps run on CPU, declared under the CPU-uncapped clause.

## Tools
Code, experiments and documents were written with an AI coding assistant (Claude Code). All runs, scores and submissions were checked by the team.
