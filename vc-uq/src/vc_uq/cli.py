"""Command line entry point.

  python -m vc_uq step1        temperature sweep only (the cheapest signal)
  python -m vc_uq gate         Phase 0 instrument validation
  python -m vc_uq generate     Phase 1 sampling into the append-only cache
  python -m vc_uq survival     Phase 3
  python -m vc_uq descriptive  Phase 2
  python -m vc_uq clm          Phase 4
  python -m vc_uq invariance   Phases 5.2-5.6 and Phase 6
  python -m vc_uq all          everything, in the protocol's build order
  python -m vc_uq smoke        end-to-end on the simulated backend

Any knob can be overridden inline:

  python -m vc_uq all -c config/default.yaml -o model.backend=llamacpp \\
      -o generation.n_max=40
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .config import load_config
from .store import Store


def _state(cfg, *, load: bool = True, run_id: str | None = None,
           new_run: bool = False):
    from .pipeline import PipelineState
    # Per-phase commands chain onto the most recent run directory for this
    # run.name, so `generate` then `gate` then `survival` land in one place.
    store = Store(cfg, run_id=run_id, attach=not new_run and run_id is None)
    state = PipelineState(cfg=cfg, store=store)
    if load:
        from .datasets import build_questions
        try:
            state.questions = store.read_questions()
        except FileNotFoundError:
            state.questions = build_questions(cfg)
        try:
            state.answers = store.read_answers()
        except FileNotFoundError:
            state.answers = None
    return state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vc_uq", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["step1", "gate", "generate", "survival",
                                        "descriptive", "clm", "invariance", "all",
                                        "smoke", "pitfalls", "runs"])
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("-o", "--override", action="append", default=[],
                    help="key.path=value, parsed as YAML")
    ap.add_argument("--labels", default=None,
                    help="CSV of hand labels with a correct_human column (Phase 0)")
    ap.add_argument("--simulate-labels", action="store_true",
                    help="use known truth instead of hand labels; simulated backend only")
    ap.add_argument("--run-id", default=None,
                    help="write into this existing run directory instead of the latest")
    ap.add_argument("--new-run", action="store_true",
                    help="start a fresh timestamped run directory for a per-phase command")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.override)
    cfg.ensure_dirs()

    from . import pipeline

    if args.command == "runs":
        from .store import run_summaries
        table = run_summaries(cfg)
        print(table.to_string(index=False) if len(table) else "no runs yet")
        return 0

    if args.command == "smoke":
        cfg = cfg.with_overrides([
            "run.name=smoke", "model.backend=mock", "embedding.backend=mock",
            "nli.backend=mock", "generation.n_max=12", "generation.r_pre=4",
            "dataset.triviaqa.n_questions=60", "dataset.fabricated.n_questions=20",
            "phase0.n_hand_label=120", "phase0.null_band.n_mismatched_pairs=600",
            "phase2.bootstrap.n_resamples=200",
            "phase4.grid.lambda_qual.num=4", "phase4.grid.lambda_div.num=3",
            "phase4.grid.lambda_stop.num=6",
            "phase5.temperature_sweep.n_questions=15",
            "phase5.temperature_sweep.n_draws=8",
            "phase5.temperature_sweep.temperatures=[0.0, 0.8, 1.5]",
            "phase5.forced_decode.n_draws=5",
        ])
        store = Store(cfg, run_id=args.run_id)
        state = pipeline.run_all(cfg, simulate_labels=True, store=store)
        print(json.dumps({"run_dir": str(store.run_dir), "notes": state.notes,
                          "pitfalls": state.results.get("pitfalls")}, indent=2,
                         default=str))
        return 0

    if args.command == "all":
        store = Store(cfg, run_id=args.run_id)
        state = pipeline.run_all(cfg, simulate_labels=args.simulate_labels,
                                 store=store)
        print(json.dumps({"run_dir": str(store.run_dir),
                          **state.results.get("pitfalls", {})}, indent=2, default=str))
        return 0

    # `generate` opens a run; the rest attach to it.
    try:
        state = _state(cfg, run_id=args.run_id,
                       new_run=args.new_run or args.command == "generate")
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.command == "generate":
        state = pipeline.step3_generate(state)
    elif args.command == "step1":
        state = pipeline.step1_temperature(state)
    elif args.command == "gate":
        if args.labels:
            labels = pd.read_csv(args.labels)
            store = state.store
            store.write_table("phase0_label_sheet_filled", labels)
        state = pipeline.step2_gate(state, simulate_labels=args.simulate_labels)
    elif args.command == "survival":
        state = pipeline.step4_survival(state)
    elif args.command == "descriptive":
        state = pipeline.step5_descriptive(state)
    elif args.command == "clm":
        state = pipeline.step6_clm(state)
    elif args.command == "invariance":
        state = pipeline.step7_invariance_transfer(state)
    elif args.command == "pitfalls":
        from .pitfalls import run_checks
        report = run_checks(cfg, answers=state.answers, questions=state.questions)
        state.store.write_table("pitfalls", report.to_frame())
        print(report.to_frame().to_string(index=False))
        return 0 if report.ok else 1

    state.store.finalize(status=f"{args.command} complete", command=args.command,
                         extra={"notes": state.notes})
    print(json.dumps({"run_dir": str(state.store.run_dir),
                      "notes": state.notes}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
