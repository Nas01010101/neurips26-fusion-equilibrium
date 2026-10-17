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
vendor/starter official loaders, errata fixes, scorer (pinned commit, gitignored)
docs/RULES.md rules and deadlines     docs/STRATEGY.md plan and risks
docs/REPRODUCE.md pinned inputs, entrypoints, compute   docs/weights.sha256 DIII-D ridge hash
```

## Commands
```
make vendor        # clone the starter kit at the pinned commit
make setup && make test
make data-train    # DIII-D train split -> /Volumes/PortableSSD
make score ARGS="--n-shots 20 --skip 7000"
```
Full pipeline (cache, train, mast, submission): docs/REPRODUCE.md. License: MIT.

Deadlines: Phase 1 ends Oct 18, Phase 2 (blind, 3 submissions) Oct 19-26.
