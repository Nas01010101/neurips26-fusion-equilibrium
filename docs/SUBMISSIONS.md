# Codabench submissions (#17739)

Scores from the public leaderboard API (`/api/phases/29580/get_leaderboard/`). Development phase, public test fold.

| Date (UTC) | File | Codabench id | DIII-D S | R²ψ | R² q95/βN | D_LCFS | Consistency | MAST S | MAST R²ψ | MAST R² q95/βN | MAST D_LCFS | MAST consistency | G |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-10 | `submission_pointer.zip` | 974601 | 0.9910 | 0.9997 | 0.9940 | 0.0133 | 0.9671 | 0.0 | −0.6644 | −5.5356 | 1.0 | 0.0 | 0.0 |
| 2026-10-15 | `submission_pointer_c3.zip` (HF rev `72d9ecbc`) | pending | | | | | | | | | | | |
| 2026-10-16 | `submission_pointer_c4.zip` (HF rev `89d03297`) | 993755 | 0.9910 | 0.9997 | 0.9940 | 0.0133 | 0.9671 | 0.7352 | 0.9791 | 0.4061 | 0.1907 | 0.2740 | 0.7418 |
| 2026-10-17 | `submission_pointer_c6.zip` (HF rev `262e20e6`), ridge DIII-D | 995984 | 0.9048 | 0.9925 | 0.7993 | 0.0330 | 0.7119 | 0.6563 | 0.8783 | 0.3485 | 0.2011 | 0.2052 | 0.7253 |
| 2026-10-17 | `submission_pointer_c7v.zip` (HF rev `0da927e1`), ridge DIII-D | not submitted | | | | | | | | | | | |
| 2026-10-17 | `submission_pointer_c7sv.zip` (HF rev `99d5d516`), ridge DIII-D | 996024 | 0.9048 | 0.9925 | 0.7993 | 0.0330 | 0.7119 | 0.6146 | 0.8827 | 0.1586 | 0.2218 | 0.1377 | 0.6793 |
| 2026-10-17 | `submission_pointer_c6b13.zip` (HF rev `ec64d00e`), ridge DIII-D, c6 with betaN window 13 | not submitted | | | | | | | | | | | |
| 2026-10-17 | `submission_pointer_c8.zip` (HF rev `d1b27e12`), ridge DIII-D, c6 with solenoid scale fixed at 0.5, betaN window 13 | not submitted (Dev closed) | | | | | | | | | | | |

- 974601 predates the MAST physics solver (`0f87b44`).
- c3 on the 3 MAST demo shots (`fusion_eq.mast_score`): S 0.579, R²ψ 0.794, R² q95/βN 0.132, 1−D_LCFS 0.851, consistency 0.188.
- c4 (`7c2c0f2`) on the same demo shots: S 0.826, R²ψ 0.995, R² q95/βN 0.664, 1−D_LCFS 0.924, consistency 0.434. Same DIII-D models as c3.
- Leaderboard on 2026-10-15, 140 entries: top DIII-D S 0.997; top G 0.94–0.98 excluding one entry with MAST S 1.000.
- c6 and c7sv choose the solenoid scale per shot (0.5, 0.75 or 1.0) against the Thomson inner edge. Shots that chose 0.5 match c4; the 11 lowest-R²ψ shots in c6's detailed result all chose 1.0. c8 fixes the scale at 0.5.
- c7sv vs c6: R²ψ +0.004, q95 R² 0.457 → 0.077, consistency 0.205 → 0.138. betaN R² is identical (0.240189), since betaN depends on inputs only.
