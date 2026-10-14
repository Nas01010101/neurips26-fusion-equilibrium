# Strategy

## Where the points are
| Term | Weight | Lever |
|---|---|---|
| R2_psi | 0.55 | Conv decoder / PCA + nonlinear head on full DIII-D train (7,041 shots) |
| Consistency | 0.20 | Axis, kappa, delta, V, li are read off the predicted psi: pixel-accurate psi near the axis and LCFS matters more than global R2 |
| R2_{q95,betaN} | 0.15 | Own heads; betaN needs pressure, so Thomson features help |
| D_LCFS | 0.10 | Boundary contour from psi; same lever as Consistency |

## Plan (ordered by risk reduction)
1. Reproduce the starter baseline (~0.81) with `make score`. Confirms data, scorer, and submission path before any modeling.
2. Beat 0.85 on DIII-D: it opens Challenge 2 and is the main score. Try Ridge -> MLP on PCA coefficients -> conv decoder, with Thomson features and aux heads (li, axis) as extra supervision.
3. Cross-machine: shape-normalized psi targets, machine-agnostic features (`features.py`), a shared physics feature space. Hold MAST out entirely for tuning except the public-test leaderboard.
4. Freeze a reproduction package before Oct 19 (pinned commit, weights, entrypoints).

## Budget
- Phase 1 has 100 submissions: spend ~1 on baseline, use local scoring for iteration, reserve submissions for confirmed local gains.
- Phase 2 gives 3 shots; the final choice should be a model that held up locally on held-out shots, not the top public-leaderboard score.

## Risks
- Leaderboard field size unknown (Codabench page not readable here). Check it manually before investing.
- G is a ratio: a high DIII-D score with a collapsed MAST score loses Challenge 2. Optimize them jointly after the gate.
- The 16 GB Mac cannot train on the full corpus in memory: stream shards from the SSD, cache features.
