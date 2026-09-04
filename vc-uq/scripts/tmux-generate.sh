#!/usr/bin/env bash
# Both shards in one tmux session, one window per GPU.
#
#   bash scripts/tmux-generate.sh          # start and attach
#
# Detach Ctrl-b d | reattach `tmux attach -t vcgen` | switch window Ctrl-b n
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SESSION="${SESSION:-vcgen}"
tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
  exit 1; }

PASS_ENV="RUN_NAME=$RUN_NAME N_SHARDS=$N_SHARDS RUN_ID=$RUN_ID${SPLITS:+ SPLITS=$SPLITS}"
tmux new-session -d -s "$SESSION" -n gpu0 -c "$REPO"
tmux send-keys -t "$SESSION:gpu0" "GPU=0 $PASS_ENV bash scripts/generate.sh" C-m
tmux new-window -t "$SESSION" -n gpu1 -c "$REPO"
tmux send-keys -t "$SESSION:gpu1" "GPU=1 $PASS_ENV bash scripts/generate.sh" C-m
echo "run id $RUN_ID -- attaching to '$SESSION'"
tmux attach -t "$SESSION"
