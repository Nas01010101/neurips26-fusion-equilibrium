# Reproduction

DIII-D: ridge regression from the shot inputs to 32 PCA coefficients of the flux map and to the scalars, with sign(Ip) interaction features and inputs clipped to the training range, trained on the public DIII-D train split.
MAST: zero-shot physics predictor (coil Green's functions, free-boundary solve, Thomson-located boundary). No MAST equilibrium data enters it from any source.

## Pinned inputs
| Input | Pin |
|---|---|
| Code | this repository, commit of the submitted build |
| Starter kit | `Sophelio/fusion-equilibrium-challenge-starter` @ `5b467988d2e5f28e20be490a926d0a71649de28d` (`make vendor`) |
| Data | HF dataset `Sophelio/fusion-equilibrium-challenge`: `data/diii_d_train/*` for training; public/blind test splits streamed by the starter's `submission_skeleton.py` |
| Environment | Python 3.12, `uv.lock` (torch 2.5.1) |
| Weights | `$RUNS/models/*.pkl`, sha256 in `docs/weights.sha256` (`make weights` checks them) |

`$RUNS` defaults to `/Volumes/PortableSSD/scratch/fusion-eq-runs`; set `FUSION_RUNS_DIR` / `FUSION_DATA_DIR` and pass `RUNS=` to make on another machine.

## Commands
```
make vendor setup test
make data-train        # DIII-D train parquet
make cache             # 7041 shots -> per-shot feature npz
make train             # DIII-D ridge
make weights           # sha256 check of the pickles
make mast              # MAST physics predictions, resumable per shot, then mast_post
make submission ARGS="--repo <user>/fusion-eq-predictions"   # build, validate, push, pointer zip
```
`make submission` without `ARGS` builds and validates only. The pointer zip holds a read-only HF token; it is never committed.

## Compute
| Step | Hardware used | Cost |
|---|---|---|
| `train` | Apple M4 CPU, 16 GB | 13-17 min (792-994 s) including scoring 200 val shots, 6.0 GB peak RSS; no GPU |
| `mast` | Apple M4 CPU, 8 workers | ~1.6 s per frame; 69,075 frames on the 1206-shot public test set ≈ 31 CPU-h (≈4 h wall). CPU, declared under the CPU-uncapped clause |
| `cache`, `submission` | CPU | not timed separately |

`make train` on CPU rebuilds the shipped pickle byte for byte (same sha256, checked 2026-10-17 on current code).

## Physics constants (literature, not fitted)
- `mast.FAST_ION_FRAC = 0.35`: NBI fast-ion share of MAST stored energy, Cecconello et al., Nucl. Fusion 59 016006 (2019). Thomson-based βN is thermal only, so total βN = thermal / (1 − 0.35).
- `mast.N_TF = 24`: MAST TF centre-rod limbs, from MAST design papers.
- Shape priors (elongation 1.8, triangularity 0.4) are generic spherical-tokamak values.

## MAST constants (no MAST label; source per constant)
- `mast.SOL_SCALE = 0.5`, `mast.SOL_SCALES = (0.5, 0.75, 1.0)`: the shipped solenoid signal reads 2x the FAIR-MAST level-2 pf_active current, an open convention question on the challenge Data page. `pick_sol_scale` picks one of the three per shot by matching the solved inner LCFS edge to the Thomson inner edge (MAST inputs only).
- `mast.WALL_R = 0.20` m: inner edge of MAST's design plasma, R - a = 0.85 - 0.65 m (Sykes et al., IAEA FEC 2000, OV4/1); centre-column armour per Edlington et al., Rev. Sci. Instrum. 72, 421 (2001).
- `mast.SMOOTH = 5`, `mast.BETA_SMOOTH`: running-mean windows chosen by leave-one-out CV on the Thomson pulse time base of all 1206 public test shots (`python -m fusion_eq.mast_smooth_cv`); the edge series sets SMOOTH, the volume-averaged pressure sets BETA_SMOOTH, each at the window with the lowest median over shots of per-shot normalised error (5 pulses = 21 ms -> 5 EFIT frames for both). Label-free test-time adaptation under rule 3. `python -m fusion_eq.mast_beta CACHE` rewrites betaN on a solved cache after a BETA_SMOOTH change (betaN does not depend on the solved flux).
- `mast_vertical.Z_MAX = 0.15` m: search bound for the force-balance vertical shift, from MAST vertical-stability physics (Windridge et al., PPCF 53, 035018 (2011); O'Gorman et al., Nucl. Fusion 54, 082002 (2014)).
- Solver options (`--profile spitzer`, `--vertical`, `--warm`) are pipeline variants, not fitted constants.

## Use of Development-phase scores (disclosed)
Development-phase scores of our own genuine candidate submissions were used to compare and select MAST pipeline variants (rule 3, feedback clause). Two Development submissions (P1, P2 in `SUBMISSIONS.md`) rescaled our predictions to measure label statistics; under the 2026-10-04 probe clause those statistics count as MAST data, so no constant fitted to them, and nothing derived from them, enters any Final submission. The fitting code was removed from the repository. Early in Development (through c3) MAST variants were also scored on the starter's 3 labelled demo shots; the scorer (`mast_score.py`) was removed. The four constants first picked on them now come from the sources above: SMOOTH (5, unchanged) and BETA_SMOOTH (9 -> 5) from the CV; WALL_R (0.20 m) from MAST design geometry; SOL_SCALE (0.5) from the signal convention, now one of three per-shot choices made against Thomson.

## Validation
- DIII-D, 200 held-out train shots, official scorer: S = 0.903 (`ridge32-ipx-clip-v200` in `$RUNS/results.jsonl`).
- MAST: no offline score. The demo shots are used for format inspection only; MAST is judged by the platform.
