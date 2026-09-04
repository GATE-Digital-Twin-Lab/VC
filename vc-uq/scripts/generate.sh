#!/usr/bin/env bash
# Phase 1 generation for one GPU.
#
#   GPU=0 bash scripts/generate.sh          # shard 0 of 2, on cuda:0
#   GPU=1 bash scripts/generate.sh          # shard 1 of 2, on cuda:1
#
# Run both, one per shell. One 27B instance per card beats one split across
# both: llama.cpp's default layer split runs the halves in sequence with a
# transfer between them.
#
# Resumable. Re-running after a crash regenerates only the missing draws --
# every cache key is a hash of (draw_set, q_id, draw_idx, T, variant), so the
# missing set is computed without calling the model. Just run it again.
#
# Optional:
#   SPLITS=tau_select   generate only that split (stages the Phase 0 gate)
#   N_SHARDS=2          how many workers share the work
#   CHECKPOINT_EVERY=500  draws between cache flushes; lower = less at risk
#   DRY_RUN=1           print the command, run nothing

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_gpu

ARGS=("${OPTS[@]}" --shard "$GPU/$N_SHARDS" --run-id "$RUN_ID")
[ -n "${SPLITS:-}" ] && ARGS+=(--splits "$SPLITS")

LOG="logs/generate_gpu${GPU}_${RUN_ID}.log"
echo "run id   : $RUN_ID   (shared via $RUNID_FILE)"
echo "shard    : $GPU/$N_SHARDS on cuda:$GPU"
echo "cache    : data/raw/${RUN_NAME}__answers.parquet"
echo "log      : $LOG"
echo

CUDA_VISIBLE_DEVICES="$GPU" run "$PY" -m vc_uq generate "${ARGS[@]}" 2>&1 | tee "$LOG"
