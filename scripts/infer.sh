#!/usr/bin/env bash
# Inference entrypoint: regenerates one Final's diii_d_private_test.npz and mast_private_test.npz
# from the Final inputs and a weights directory.
#   scripts/infer.sh 1|2|3 [WEIGHTS_DIR]     WEIGHTS_DIR defaults to weights/
# Output: $FUSION_RUNS_DIR/infer/final<N>/, with sha256 next to the submitted files' hashes;
# SUBMITTED_DIR=<dir with the submitted .npz> also compares them array by array.
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:?usage: scripts/infer.sh 1|2|3 [WEIGHTS_DIR]}"
W="$(cd "${2:-weights}" && pwd)"
export PYTHONPATH=src FUSION_SPLIT=private_test
: "${FUSION_RUNS_DIR:=/Volumes/PortableSSD/scratch/fusion-eq-runs}"
: "${WORKERS:=8}"
export FUSION_RUNS_DIR
OUT="$FUSION_RUNS_DIR/infer"
mkdir -p "$OUT"
PY="uv run python -m"
step() { echo "== $(date -u +%FT%TZ) $*"; }

# DIII-D file, shared by all three Finals: ridge, then q95/betaN shrink at 0.4
step "DIII-D"
mkdir -p "$FUSION_RUNS_DIR/models" && cp "$W/ridge32-ipx-clip-v200.pkl" "$FUSION_RUNS_DIR/models/"
# with --final the starter builds both Final configs, so the MAST one stops on the empty MAST cache
# after the DIII-D file is written; only that file is kept
[ -f "$OUT/d3d/diii_d_private_test.npz" ] || $PY fusion_eq.make_submission --final --max-shots 0 --out "$OUT/d3d" \
  || [ -f "$OUT/d3d/diii_d_private_test.npz" ]
[ -f "$OUT/d3d_s04/diii_d_private_test.npz" ] || { mkdir -p "$OUT/d3d_s04"; $PY fusion_eq.d3d_shrink "$OUT/d3d/diii_d_private_test.npz" "$OUT/d3d_s04/diii_d_private_test.npz" --lam 0.4; }

# MAST: solve, optional closure re-solve, twin correction, betaN window, vertical shift
step "MAST solve"
[ -d "$OUT/ROWS" ] || $PY fusion_eq.dump_rows "$OUT/ROWS"
$PY fusion_eq.mast_batch --sol 0.5 --freeb-iter 4 --rows "$OUT/ROWS" --cache "$OUT/IT4" --workers "$WORKERS"
case "$N" in
  1) BASE=CL; TW=twin-mast-mode.pkl ;;
  2) BASE=IT4; TW=twin-mast-mode.pkl ;;
  3) BASE=CL; TW=twin-mast-b75.pkl ;;
  *) echo "Final must be 1, 2 or 3" >&2; exit 2 ;;
esac
if [ "$BASE" = CL ]; then
  step "closure solve"
  $PY fusion_eq.mast_batch --sol 0.5 --freeb-iter 4 --rows "$OUT/ROWS" --sol-from "$OUT/IT4" --closure "$W/closure-d3d-gbm.pkl" --cache "$OUT/CL" --workers "$WORKERS"
fi
T="$OUT/T$N"
step "twin, betaN, vertical"
[ -d "$T" ] || $PY fusion_eq.twin apply --cache "$OUT/$BASE" --rows "$OUT/ROWS" --model "$W/$TW" --c 1.0 --out "$T" --workers "$WORKERS"
[ "$N" = 2 ] && $PY fusion_eq.mast_beta "$T" --rows "$OUT/ROWS" --win 13   # Finals 1 and 3 keep BETA_SMOOTH 5
[ -d "${T}V" ] || $PY fusion_eq.mast_vertical "$T" "${T}V" --rows "$OUT/ROWS"
step "final files"
rm -rf "$OUT/final$N"
$PY fusion_eq.swap_mast "$OUT/d3d_s04" "$OUT/final$N" "${T}V"
(cd "$OUT/final$N" && shasum -a 256 *.npz)
grep -A2 "final$N" docs/SUBMITTED.sha256 2>/dev/null || true
# the DIII-D ridge output can differ from the submitted file at float32 rounding (~5e-7), which
# changes its sha256; with the submitted files (HF revision in docs/SUBMITTED.sha256) in
# $SUBMITTED_DIR, compare arrays instead
if [ -n "${SUBMITTED_DIR:-}" ]; then
  for f in diii_d_private_test.npz mast_private_test.npz; do
    echo "$f"; uv run python scripts/npzcmp.py "$SUBMITTED_DIR/$f" "$OUT/final$N/$f"
  done
fi
