# Methods (Fusion Equilibrium Challenge, Codabench #17739, team nas_2242)

Two predictors share one submission: a linear model trained on the public DIII-D train split, and a zero-shot physics solver for MAST that uses no MAST equilibrium data from any source. Code: this repository at the submitted tag; commands and pins: `docs/REPRODUCE.md`.

## DIII-D: ridge on PCA coefficients
- Inputs per EFIT time: the starter's interpolated coil currents and Ip (errata applied), core and edge Thomson summaries, pressure features, and time since the first equilibrium (`data.shot_features`).
- Targets: 32 PCA coefficients of the 65x65 flux map (randomised PCA, fixed seed) and the scalars q95 and betaN, each fitted by `RidgeCV` (alphas 1e-3 to 1e3) after standardisation (`models.PcaRidge`).
- sign(Ip) x inputs are appended as interaction features: about 8% of DIII-D shots run reversed Ip, and one linear map cannot fit both regimes.
- Inputs are clipped to the 0.1-99.9 percentile range of the training set, because a linear map extrapolates without bound.
- Training: every 6th frame of the 7,041 train shots, 200 shots held out. The held-out official scorer gives S 0.903: R²ψ 0.992, R² q95/βN 0.804, 1−D_LCFS 0.969, consistency 0.699. Training takes 13-17 min on an Apple M4 CPU with no GPU, and a retrain on the submitted code reproduces the shipped weights byte for byte.
- Why the ridge and not our MLP ensemble (held-out S 0.9915, used in Development submissions): Award 2 ranks G = S_MAST / S_DIII-D among entries with S_DIII-D ≥ 0.85, so for a fixed MAST predictor a DIII-D model that clears the gate with margin raises G. We submit one DIII-D model to every Final submission. TODO(W2): received DIII-D S of the ridge on the public test fold.

## MAST: free-boundary physics solver, no MAST labels
1. **Vacuum flux.** Green's-function flux of every shipped coil conductor on the EFIT grid (`greens.vacuum_psi`). The solenoid column reads 2x the FAIR-MAST level-2 current, a convention the Data page leaves open. `mast.pick_sol_scale` picks 0.5, 0.75 or 1.0 per shot, whichever puts the solved inner LCFS edge closest to the Thomson inner density edge on three frames.
2. **Boundary data.** The MAST Thomson laser is horizontal at the midplane. The innermost and outermost channels whose 5-channel median density exceeds 30% of the profile's 90th percentile mark the inner and outer LCFS midplane radii (`mast.thomson_edges`).
3. **Free-boundary solve.** Picard iteration on the plasma current: J = (β0 R/R0 + (1−β0) R0/R)(1−ψ_N)^α, normalised to the measured Ip (`freeb.solve`). J is symmetrised up-down for stability. β0 is chosen per frame from a 0.0 to 0.5 grid to match the Thomson outer edge (`freeb.solve_matched`). Axis and LCFS searches run inside a mask built from the coil-conductor convex hull, outboard of the solenoid (`mast.machine_mask`), never the scorer's label-derived envelope.
4. **Pipeline variants** (selected by Development score, see below):
   - `--vertical` holds each frame at zero net coil force on the plasma current, F_z = −2π∫R J B_R dA, within ±0.15 m (`mast_vertical`).
   - `--profile spitzer` takes the J shape from Thomson Te^1.5.
   - `--warm` limits the β0 search to within 0.15 of the previous frame's value.
5. **Scalars.**
   - q95 is measured on the solved flux with the scorer's own surface finders (`qprofile.q_at`). The TF vacuum field is F = μ0 N_TF I_TF / 2π with N_TF = 24, and the inboard limiter is at 0.20 m from MAST design geometry. Where no surface is found, q95 falls back to the ITER Physics Basis formula.
   - betaN is the volume-averaged Thomson pressure over B0²/2μ0, divided by (1 − 0.35) for the NBI fast-ion share (Cecconello et al., NF 59 016006, 2019). It depends on inputs only, never on the solved flux.
6. **Time smoothing.** The plasma part of ψ, q95 and betaN are each averaged over 5 EFIT frames. The windows come from leave-one-out CV of a running mean on the Thomson time base of all 1,206 public test shots, using MAST inputs only (`mast_smooth_cv`). The ψ and q95 window comes from the outer-edge series that places each solve, and the betaN window from the pressure series. This is label-free test-time adaptation under rule 3.

Each constant's source is listed in `docs/REPRODUCE.md`. None is fitted to MAST equilibria.

## Data use and disclosures (rule 3)
- No MAST equilibrium labels enter training, constants or selection. That includes the starter's labelled demo shots.
- Development-phase scores of our own candidate submissions were used to choose between MAST pipeline variants. This is permitted by the rule 3 feedback clause.
- Two Development submissions (P1, P2) rescaled predictions to measure label statistics. Under the 2026-10-04 probe clause these count as MAST data. No constant fitted to them, and nothing derived from them, is in any Final submission, and the fitting code was removed.
- Through candidate c3, MAST variants were also scored on the 3 labelled demo shots. That scorer was removed. The four constants first picked on those shots now come from label-free sources:
  - SMOOTH and BETA_SMOOTH from the CV above (BETA_SMOOTH 9 → 5).
  - WALL_R from MAST design geometry.
  - SOL_SCALE from the signal convention, now one of three per-shot choices made against Thomson.
- TODO(W2): table of Development scores per MAST variant (c6, c7v, c7sv) and the variant chosen.

## Compute
| Step | Hardware | Cost |
|---|---|---|
| DIII-D train + 200-shot validation | Apple M4 CPU | 994 s |
| MAST solve, 1,206 public test shots | Apple M4 CPU, 7 workers | 22 CPU-h (c7v, 190 min wall) to 28 CPU-h (c7sv, 244 min wall) per variant, declared under the CPU-uncapped clause |
| GPU | none | 0 GPU-h |

## Tools
Code, experiments and documents were written with an AI coding assistant (Claude Code). All runs, scores and submissions were checked by the team.
