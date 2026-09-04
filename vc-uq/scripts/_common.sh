# Shared settings for the vc-uq run scripts. Sourced, not executed.
#
# The short-answer arm is the study's primary (see protocol section 3 and the
# Phase 0 comparison): constraining answer form removes a length confound that
# put e_cos at r = 0.80 with answer length among human-correct answers, and
# moved tau_star from 0.91 -- effectively on top of the mismatched-pair null --
# down to 0.37.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY (set PY=...)" >&2; exit 1; }

RUN_NAME="${RUN_NAME:-vc_short}"
N_SHARDS="${N_SHARDS:-2}"

# Every knob the short arm needs, in one place. Override any of them from the
# environment; add more with EXTRA="-o key=value".
OPTS=(
  -o "run.name=$RUN_NAME"
  -o "generation.prompt_variant_post=${VARIANT_POST:-vc_post_short_v1}"
  -o "generation.prompt_variant_clean=${VARIANT_CLEAN:-answer_clean_short_v1}"
  -o "generation.checkpoint_every=${CHECKPOINT_EVERY:-500}"
)
[ -n "${EXTRA:-}" ] && OPTS+=($EXTRA)

mkdir -p logs

# Both shards must land in ONE run directory, so the run id is shared through a
# file rather than computed per process -- two `date` calls a second apart give
# two directories and split the manifest. `set -o noclobber` makes the create
# atomic, so if both shards start together the loser simply reads the winner's
# value. Delete the file to begin a fresh run directory.
RUNID_FILE="${RUNID_FILE:-logs/$RUN_NAME.runid}"
if [ -z "${RUN_ID:-}" ]; then
  (set -o noclobber; date +%Y%m%d-%H%M%S > "$RUNID_FILE") 2>/dev/null || true
  RUN_ID="$(cat "$RUNID_FILE")"
fi

require_gpu() {
  [ -n "${GPU:-}" ] || {
    echo "GPU is required, e.g.  GPU=0 bash scripts/$(basename "$0")" >&2
    exit 2
  }
}

# DRY_RUN=1 prints the command instead of running it.
run() {
  if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "$@"; echo; else "$@"; fi
}
