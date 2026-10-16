# Codabench submissions (#17739)

Scores from the public leaderboard API (`/api/phases/29580/get_leaderboard/`). Development phase, public test fold.

| Date (UTC) | File | Codabench id | DIII-D S | R²ψ | R² q95/βN | D_LCFS | Consistency | MAST S | MAST R²ψ | MAST R² q95/βN | MAST D_LCFS | MAST consistency | G |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-10 | `submission_pointer.zip` | 974601 | 0.9910 | 0.9997 | 0.9940 | 0.0133 | 0.9671 | 0.0 | −0.6644 | −5.5356 | 1.0 | 0.0 | 0.0 |
| 2026-10-15 | `submission_pointer_c3.zip` (HF rev `72d9ecbc`) | pending | | | | | | | | | | | |
| 2026-10-16 | `submission_pointer_c4.zip` (HF rev `89d03297`) | pending | | | | | | | | | | | |

- 974601 predates the MAST physics solver (`0f87b44`).
- c3 on the 3 MAST demo shots (`fusion_eq.mast_score`): S 0.579, R²ψ 0.794, R² q95/βN 0.132, 1−D_LCFS 0.851, consistency 0.188.
- c4 (`7c2c0f2`) on the same demo shots: S 0.826, R²ψ 0.995, R² q95/βN 0.664, 1−D_LCFS 0.924, consistency 0.434. Same DIII-D models as c3.
- Leaderboard on 2026-10-15, 140 entries: top DIII-D S 0.997; top G 0.94–0.98 excluding one entry with MAST S 1.000.
