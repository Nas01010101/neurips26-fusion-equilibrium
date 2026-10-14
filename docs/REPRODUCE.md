# Reproduction

DIII-D: ensemble of three MLPs on PCA coefficients of the flux map, trained on the public DIII-D train split.
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
make train             # 3 ensemble members, seeds 0/1/2
make weights           # sha256 check of the pickles
make mast              # MAST physics predictions, resumable per shot
make submission ARGS="--repo <user>/fusion-eq-predictions"   # build, validate, push, pointer zip
```
`make submission` without `ARGS` builds and validates only. The pointer zip holds a read-only HF token; it is never committed.

## Compute
| Step | Hardware used | Cost |
|---|---|---|
| `train` | Apple M4 (MPS), 16 GB | ~30 min per member including scoring 200 val shots; well under 1 A100 GPU-h for all three |
| `mast` | Apple M4 CPU, 8 workers | ~1.6 s per frame; 69,075 frames on the 1206-shot public test set ≈ 31 CPU-h (≈4 h wall). CPU, declared under the CPU-uncapped clause |
| `cache`, `submission` | CPU | not timed separately |

Retraining on MPS or CUDA is not bit-identical (non-deterministic kernels); the shipped pickles are the reference.

## Physics constants (literature, not fitted)
- `mast.FAST_ION_FRAC = 0.35`: NBI fast-ion share of MAST stored energy, Cecconello et al., Nucl. Fusion 59 016006 (2019). Thomson-based βN is thermal only, so total βN = thermal / (1 − 0.35).
- `mast.N_TF = 24`: MAST TF centre-rod limbs, from MAST design papers.
- Shape priors (elongation 1.8, triangularity 0.4) are generic spherical-tokamak values.

## Validation
- DIII-D, 200 held-out train shots, official scorer: S = 0.9915 (`ens3-w2048` in `$RUNS/results.jsonl`).
- MAST, 3 public demo shots (scoring only, no tuning): S = 0.579.
