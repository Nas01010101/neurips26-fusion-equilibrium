# neurips26-fusion-equilibrium

Entry of team nas_2242 in the NeurIPS 2026 Fusion Equilibrium Challenge (Codabench #17739). For each EFIT time it predicts the poloidal flux ψ on the 65×65 grid, q95 and βN from coil currents, Ip and Thomson scattering, on two machines:

- **DIII-D:** ridge regression from the shot inputs to 32 PCA coefficients of ψ and to the two scalars, trained on the public DIII-D train split.
- **MAST:** a free-boundary Grad-Shafranov solve from coil Green's functions, with the plasma edge placed by Thomson scattering. No MAST equilibrium data enters it from any source. A residual model corrects the solve; it is trained only on synthetic equilibria from the same solver.

Method, validation and data-use disclosures: [docs/METHODS.md](docs/METHODS.md). Pinned inputs, commands and the source of every constant: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Final submissions
All three Finals carry the same DIII-D file (ridge, q95 and βN shrunk toward the training mean at 0.4). They differ in the MAST pipeline:

| Final | MAST pipeline |
|---|---|
| 1 | solve → learned current-profile closure → twin correction, per-mode scales → vertical shift |
| 2 | solve → twin correction, per-mode scales → vertical shift |
| 3 | solve → learned current-profile closure → twin correction, blended scales → vertical shift |

## Quickstart
```
make vendor          # starter kit at the pinned commit (scorer, loaders, errata)
make setup           # uv environment from uv.lock (Python 3.12)
make test            # unit tests, no data needed
make weights         # sha256 check of weights/*.pkl
```
Data and run directories default to an external volume. Set `DATA=` and `RUNS=` on the make line (and `FUSION_DATA_DIR`, `FUSION_RUNS_DIR` for direct module calls) to use other paths. The DIII-D training, MAST solve and Final build commands are in docs/REPRODUCE.md.

## Layout
```
src/fusion_eq/
  data.py, features.py, splits.py   DIII-D parquet to per-shot feature cache, shot-level folds
  models.py, evaluate.py            PCA + ridge model; training and scoring with the official scorer
  d3d_shrink.py                     q95/βN shrink of the DIII-D file
  greens.py, freeb.py, qprofile.py  coil Green's functions, free-boundary solve, q from a flux map
  mast.py, mast_batch.py            MAST predictor and the parallel per-shot runner
  mast_beta.py, mast_post.py        βN and time smoothing on a solved cache
  mast_smooth_cv.py                 label-free choice of the smoothing windows
  twin.py                           synthetic-twin residual model: gen, fit, modes, map, blend, apply
  closure.py                        current-profile exponent learned on DIII-D EFIT
  mast_vertical.py                  vertical shift from coil force balance
  d3d_bench.py                      MAST pipeline variants scored on labelled DIII-D shots
  te_check*.py, mast_bench.py       Thomson Te checks on MAST predictions (no labels)
  final_gate.py                     upload checks for a built Final
  make_submission.py, submit.py, swap_mast.py   npz writer and submission assembly
tests/          pytest suite
weights/        ridge, closure and twin pickles; hashes in docs/weights.sha256
docs/           METHODS, REPRODUCE, BENCH, SUBMISSIONS, shot lists, gate config
vendor/starter  challenge starter kit (cloned by make vendor, not tracked)
```

## Compute
No GPU. DIII-D training takes 13 to 17 minutes on an Apple M4 CPU. A MAST pipeline takes 22 to 28 CPU-hours on the 1,206 public test shots, declared under the challenge's CPU-uncapped clause.

## Licence and tools
MIT ([LICENSE](LICENSE)). Code, experiments and documents were written with an AI coding assistant (Claude Code); the team checked all runs, scores and submissions.
