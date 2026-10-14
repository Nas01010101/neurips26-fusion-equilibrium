# Fusion Equilibrium Challenge: rules (read 2026-10-07)

Sources: fusion-equilibrium-challenge.sophelio.io (+ /rules/), starter repo `Sophelio/fusion-equilibrium-challenge-starter` @ 5b46798, Codabench #17739 (page not readable without JS; leaderboard unseen).

## Task
Predict from non-magnetic inputs only: coil currents, Thomson Te/ne, Ip.
Outputs per shot, per `efit_times` timestamp: `psirz` (T,65,65), `q95` (T,), `betaN` (T,).

## Scores
`S = 0.55 R2_psi + 0.15 R2_{q95,betaN} + 0.10 (1 - D_LCFS) + 0.20 Consistency`
- Challenge 1 ($500): best S on DIII-D.
- Challenge 2 ($500): best `G = S(MAST) / S(DIII-D)`; gate `S(DIII-D) >= 0.85`, else G = 0. Must submit both machines. MAST: zero training data, and no MAST equilibrium data from any source.
- Starter baseline reaches ~0.81 composite on DIII-D, so the 0.85 gate sits above it.

## Dates and limits
- Phase 1: Jul 27 - Oct 18. 5 submissions/day, 100 total, public test fold.
- Phase 2: Oct 19 - Oct 26. Blind test, 3 submissions, leaderboard hidden. Reproduction package mandatory for winners.
- Reproduction: pinned commit, weights, entrypoints, data provenance; trainable in <= 24 GPU-h on an 80 GB A100-class GPU (CPU work uncapped if declared). Top 3 release code under an OSI license and send a 1-2 page methods report before payment.

## Allowed
External public datasets (other tokamak archives, OMFIT tables) and public pretrained models, with disclosure. Any language or framework.

## Pitfalls from the starter
- Never round/truncate/decimate predictions (`np.round(psi, 3)` costs ~35% of MAST Consistency).
- Resample inputs onto `efit_times`; never interpolate targets.
- DIII-D `magnetics_plasma_current_times` is wrong on ~69% of shots: use `data_fixes.fix_d3d_ip_times`. MAST Thomson core has 130 R vs 131 channels: also in `data_fixes.py`.
- Split by shot. DIII-D and MAST flux sign conventions differ (scorer normalizes).
- Submissions are `submission_pointer.zip` pointing at a Hugging Face predictions repo (`submission_skeleton.py`, `push_predictions.py`).

## Unknown
Team size limit, tie rules, public leaderboard state.
