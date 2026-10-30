#!/usr/bin/env bash
# Training entrypoint: rebuilds every shipped weight file from the public DIII-D train split and the
# Final MAST inputs (no MAST labels), then compares each against weights/ by sha256.
#   scripts/train.sh            outputs under $FUSION_RUNS_DIR/rebuild
# Steps skip when their output exists, so a stopped run resumes. Commands follow docs/REPRODUCE.md.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
: "${FUSION_DATA_DIR:=/Volumes/PortableSSD/datasets/fusion-equilibrium}"
: "${FUSION_RUNS_DIR:=/Volumes/PortableSSD/scratch/fusion-eq-runs}"
: "${WORKERS:=8}"
export FUSION_DATA_DIR FUSION_RUNS_DIR
OUT="$FUSION_RUNS_DIR/rebuild"
mkdir -p "$OUT/w"
PY="uv run python -m"
step() { echo "== $(date -u +%FT%TZ) $*"; }

# shot lists -> parquet paths; bench halves A = odd lines, B = even lines
for l in bench300 d3d_train; do
  sed "s|^|$FUSION_DATA_DIR/data/diii_d_train/|; s|$|.parquet|" "docs/lists/$l.txt" > "$OUT/${l}_files.txt"
done
awk 'NR % 2 == 1' "$OUT/bench300_files.txt" > "$OUT/half_a.txt"
awk 'NR % 2 == 0' "$OUT/bench300_files.txt" > "$OUT/half_b.txt"

# DIII-D ridge
step "DIII-D feature cache"
[ -d "$FUSION_RUNS_DIR/cache" ] || $PY fusion_eq.data "$FUSION_DATA_DIR/data/diii_d_train" "$FUSION_RUNS_DIR/cache"
step "DIII-D ridge"
[ -f "$OUT/w/ridge32-ipx-clip-v200.pkl" ] || {
  $PY fusion_eq.evaluate --model pca_ridge --stride 6 --n-val 200 --kw '{"ip_interact": true}' --tag ridge32-ipx-clip-v200 --save
  cp "$FUSION_RUNS_DIR/models/ridge32-ipx-clip-v200.pkl" "$OUT/w/"
}

# current-profile closure, DIII-D EFIT labels only
step "closure"
[ -f "$OUT/CLD.npz" ] || $PY fusion_eq.closure data --files "$OUT/d3d_train_files.txt" --exclude "$OUT/bench300_files.txt" --n 1500 --workers "$WORKERS" --out "$OUT/CLD.npz"
[ -f "$OUT/w/closure-d3d-gbm.pkl" ] || $PY fusion_eq.closure fit --data "$OUT/CLD.npz" --model "$OUT/w/closure-d3d-gbm.pkl"

# MAST twin: synthetic equilibria from the Final inputs, solver output as the only target
export FUSION_SPLIT=private_test
step "MAST rows and base solve"
[ -d "$OUT/ROWS" ] || $PY fusion_eq.dump_rows "$OUT/ROWS"
$PY fusion_eq.mast_batch --sol 0.5 --freeb-iter 4 --rows "$OUT/ROWS" --cache "$OUT/IT4" --workers "$WORKERS"
step "MAST twin"
[ -d "$OUT/TWM" ] || $PY fusion_eq.twin gen --machine mast --rows "$OUT/ROWS" --sol-from "$OUT/IT4" --out "$OUT/TWM" --frames 8 --workers "$WORKERS"
[ -f "$OUT/TWM.pkl" ] || $PY fusion_eq.twin fit --data "$OUT/TWM" --model "$OUT/TWM.pkl" --kind ridge

# per-mode scales from the DIII-D bench, carried to the MAST twin
step "DIII-D twin and bench"
[ -d "$OUT/TWD" ] || $PY fusion_eq.twin gen --machine d3d --files "$OUT/bench300_files.txt" --out "$OUT/TWD" --frames 8 --draws 3 --workers "$WORKERS"
[ -f "$OUT/TWD.pkl" ] || $PY fusion_eq.twin fit --data "$OUT/TWD" --model "$OUT/TWD.pkl" --kind ridge
$PY fusion_eq.d3d_bench --files "$OUT/bench300_files.txt" --variant iter4 --out "$OUT/BENCH" --workers "$WORKERS"
FUSION_TWIN_MODEL="$OUT/TWD.pkl" $PY fusion_eq.d3d_bench --files "$OUT/bench300_files.txt" --variant twin --out "$OUT/BENCH" --workers "$WORKERS"
step "mode scales"
$PY fusion_eq.twin modes --model "$OUT/TWD.pkl" --files "$OUT/half_a.txt" --bench "$OUT/BENCH" --out "$OUT/MS_A.pkl"
$PY fusion_eq.twin modes --model "$OUT/TWD.pkl" --files "$OUT/half_b.txt" --bench "$OUT/BENCH" --out "$OUT/MS_B.pkl"
$PY fusion_eq.twin map --from "$OUT/MS_A.pkl" "$OUT/MS_B.pkl" --model "$OUT/TWM.pkl" --out "$OUT/w/twin-mast-mode.pkl"
# global scale c, fixed at 0.39 from half A before the Final (first in docs 2026-10-23; half B gave 0.397); printed here as a check
uv run python scripts/global_scale.py "$OUT/half_a.txt" "$OUT/BENCH"
uv run python scripts/global_scale.py "$OUT/half_b.txt" "$OUT/BENCH"
$PY fusion_eq.twin blend --model "$OUT/w/twin-mast-mode.pkl" --lam 0.75 --c 0.39 --out "$OUT/w/twin-mast-b75.pkl"

step "compare with weights/"
(cd "$OUT/w" && shasum -a 256 -c "$OLDPWD/docs/weights.sha256") || echo "rebuilt weights differ from weights/ (compare scores with scripts/infer.sh)"
