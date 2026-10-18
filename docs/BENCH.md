# Bench A: selecting MAST changes on labelled DIII-D shots

Written 2026-10-18, before any Bench A variant was scored.

## Why

MAST has no labels we may use (Rule 3). The label-free Thomson checks in `fusion_eq.mast_bench`
rank the Development-phase MAST submissions in the opposite order to the server: on the shots the
server scored worst for c6 and c7sv, they rated those variants better than c4 (P(worse) 0.04–0.11).
They no longer select anything.

`fusion_eq.d3d_bench` runs the MAST free-boundary solver (`freeb.solve_matched`, the same shape
priors and `machine_mask`) on DIII-D training shots and scores the result with the byte-identical
`local_score` functions. The substitutions are listed in the module docstring; every variant gets
them identically.

## Set

60 `diii_d_train` shots: every 117th file in sorted order, starting at index 2, with every 10th
EFIT frame. None of them was used to fit `E_SCALE`.

## Decision rule

A change ships in a Final MAST submission only if all of these hold:

1. On Bench A its paired shot-bootstrap 95% CI of ΔS against `sym` (the current MAST default)
   lies entirely above 0.
2. No Consistency scalar loses more than 0.05 R².
3. It does not need a constant fitted on MAST data.

A change that touches only MAST-specific inputs (Thomson profiles, the MAST solenoid convention)
cannot be tested here. It ships only with a stated physics reason, and never on a label-free
proxy score alone.

## What it cannot tell

- How the MAST vacuum (solenoid convention, EFPS, no vessel currents) behaves.
- q95 and βN: both are set to the shot's EFIT mean, so Bench A ranks ψ, D_LCFS and Consistency.
- Whether DIII-D's mostly single-null shapes stand in for MAST's double-null ones. A gain that
  comes from up-down asymmetry is the case most likely to transfer badly.
