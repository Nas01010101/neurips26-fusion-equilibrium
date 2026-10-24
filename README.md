# neurips26-fusion-equilibrium

Entry for the NeurIPS 2026 Fusion Equilibrium Challenge: reconstruct tokamak plasma flux maps (psi on a 65x65 grid, plus q95 and betaN) from coil currents, Thomson scattering and Ip.

## Layout
```
src/fusion_eq/
  paths.py     data/runs locations (SSD) and vendored starter on sys.path
  features.py  machine-agnostic features, psi shape normalization
  models.py    model registry; PCA + Ridge baseline
  splits.py    shot-level folds
  submit.py    npz writer for the Codabench contract
  d3d_shrink.py   q95/betaN shrink toward train means (DIII-D file)
  freeb.py, greens.py, mast*.py   MAST free-boundary Grad-Shafranov solve and post-steps
  mast_vertical.py   force-balance vertical shift
  twin.py      synthetic-twin correction: gen, fit, modes, map, blend, apply
  closure.py   learned current-profile exponent for the MAST solve (DIII-D labels only)
  d3d_bench.py    MAST pipeline variants scored on DIII-D train shots
  te_check_mast.py, final_gate.py   Thomson Te check and Final upload gate
  swap_mast.py    MAST predictions into a DIII-D submission
vendor/starter official loaders, errata fixes, scorer (pinned commit, gitignored)
docs/RULES.md rules and deadlines     docs/STRATEGY.md plan and risks
docs/REPRODUCE.md pinned inputs, entrypoints, compute   docs/weights.sha256 hashes of weights/*.pkl
```

## Commands
```
make vendor        # clone the starter kit at the pinned commit
make setup && make test
make data-train    # DIII-D train split -> /Volumes/PortableSSD
make score ARGS="--n-shots 20 --skip 7000"
```
Full pipeline (cache, train, mast, submission): docs/REPRODUCE.md. License: MIT.

Final phase: 3 blind submissions, closes 2026-10-26 00:00 UTC. Submitted pipelines: docs/METHODS.md.
