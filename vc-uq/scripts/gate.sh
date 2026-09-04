#!/usr/bin/env bash
# Phase 0 -- instrument validation. Needs no GPU shard; runs on one card.
#
#   GPU=1 bash scripts/gate.sh                    # writes the label sheet, then stops
#   GPU=1 LABELS=<sheet.csv> bash scripts/gate.sh # scores your hand labels
#
# Gate condition: kappa >= 0.70 AND AUROC(primary) >= 0.85 AND null-band
# separation >= 0.75. Below the AUROC bar the primary criterion switches from
# cosine to NLI automatically.
#
# Exit codes: 0 pass or fail-with-report, 1 sheet written and labels needed,
# 3 the labels do not cover every stratum (it names which).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_gpu

ARGS=("${OPTS[@]}")
[ -n "${LABELS:-}" ] && ARGS+=(--labels "$LABELS")

LOG="logs/gate_${RUN_NAME}.log"
echo "run name : $RUN_NAME"
echo "labels   : ${LABELS:-<none -- will write a sheet and stop>}"
echo "log      : $LOG"
echo

CUDA_VISIBLE_DEVICES="$GPU" run "$PY" -m vc_uq gate "${ARGS[@]}" 2>&1 | tee "$LOG"
