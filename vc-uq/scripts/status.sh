#!/usr/bin/env bash
# Generation progress. No GPU needed. WATCH=1 to refresh every 60s.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

report() {
  "$PY" - "$RUN_NAME" <<'PY'
import sys, pandas as pd
from pathlib import Path
from vc_uq.config import load_config
from vc_uq.datasets import build_questions

name = sys.argv[1]
cfg = load_config(overrides=[f"run.name={name}"])
qs = build_questions(cfg)
n_max = int(cfg.get("generation.n_max")); r_pre = int(cfg.get("generation.r_pre"))
dwn = [str(s) for s in cfg.get("generation.downstream_splits")]
target = len(qs) * n_max + int(qs.split.isin(dwn).sum()) * n_max

p = Path(f"data/raw/{name}__answers.parquet")
if not p.exists():
    print(f"{name}: no cache yet (target {target:,} draws)"); raise SystemExit
d = pd.read_parquet(p)
pct = 100 * len(d) / target
print(f"{name}: {len(d):,} / {target:,} draws  ({pct:5.1f}%)   {d.q_id.nunique()} / {len(qs)} questions")
print("  by pass :", d.draw_set.value_counts().to_dict())
print("  parse   :", d.parse_status.value_counts().to_dict())
bad = float(d.vc_post.isna().mean())
print(f"  vc unparsed: {bad:.2%}  (a pitfall check fails above "
      f"{float(cfg.get('generation.max_vc_parse_failure_rate')):.0%})")
v = Path(f"data/raw/{name}__vc_pre_repeats.parquet")
if v.exists():
    vv = pd.read_parquet(v)
    print(f"  vc_pre  : {len(vv):,} / {len(qs) * r_pre:,}")
PY
}

if [ -n "${WATCH:-}" ]; then while true; do clear; date; echo; report; sleep 60; done
else report; fi
