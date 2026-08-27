# vc-uq

Implementation of `protocol.md`: does verbalised confidence (VC) function as an
estimator of a model's per-sample correctness probability `p_q`?

The thesis under test is not that VC is uninformative — Tian et al. show it is
often better calibrated than conditional token probabilities. It is that VC
violates invariance properties any estimator of `p_q` must satisfy, that its
errors are correlated across samples of the same question so confidence-based
aggregation fails *worse* with more evidence, and that its dynamic range spans
roughly one sample of operational difference.

## State

Every phase in the protocol is implemented and runs end to end. No real
generation has been performed — the backend is pluggable and the pipeline has
been validated against a simulated model with known ground truth.

| Protocol | Module | Status |
|---|---|---|
| §2 schemas, §1 cache | `schemas.py`, `store.py` | done; per-run output dirs |
| §3 Phase 0 gate | `phase0.py` | done; needs hand labels |
| §4 Phase 1 generation | `generate.py`, `backends/` | done |
| §5 Phase 2 descriptive | `calibration.py` | done |
| §6 Phase 3 survival/U/2×2/product rule | `survival.py`, `cluster.py` | done |
| §7 Phase 4 CLM + LTT | `clm.py`, `ltt.py` | done |
| §8 Phase 5 invariance | `invariance.py` | done |
| §9 Phase 6 transfer | `transfer.py` | done |
| §11 pitfall checklist | `pitfalls.py` | done, executable |

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
python -m vc_uq smoke      # whole pipeline on the simulated backend, ~1 minute
pytest -q                  # 68 tests
```

`smoke` deliberately runs at `N_MAX = 12`, so it *fails* the KM-flatness and
non-vacuity gates — that is the checks working, not a bug.

## Output layout

Every invocation writes derived output into a timestamped run directory:

```
results/runs/20260827-094939__smoke/
    manifest.json          what ran, when, on which model, at which git rev
    config.snapshot.yaml   the fully resolved config, overrides included
    tables/                *.csv, *.json
    figures/               *.png
    processed/             *.parquet
data/raw/                  SHARED generation cache, keyed by run.name
    smoke__answers.parquet
```

The raw cache is deliberately **not** timestamped. It is the expensive artifact,
and the whole design rests on downstream phases being pure functions of it
(§1: *every phase must be re-runnable without re-sampling*). Giving each run its
own copy would mean re-sampling every time. Re-running generation against an
existing cache is a no-op on rows already present.

`all`, `smoke` and `generate` open a new run directory; every other command
attaches to the most recent one for that `run.name`, so a phase-by-phase session
lands in one place:

```bash
python -m vc_uq generate              # opens results/runs/20260827-...__default
python -m vc_uq gate --labels x.csv   # attaches to it
python -m vc_uq survival              # attaches to it
python -m vc_uq runs                  # list every run with its manifest
```

`--run-id` targets a specific directory; `--new-run` branches a fresh one.

Config-derived manifest fields are fixed when the run is created and never
restamped. If a later command carries different `-o` overrides, the manifest
records that command's own digest and sets `config_drift: true` — which is how
you find out `survival` ran under different knobs than `generate` did.

## Running against a real model

Generation is llama.cpp in-process via `llama-cpp-python`, not `llama-server`.
The reason is `h_tok`: the protocol defines it over the **full** next-token
distribution, and the HTTP API returns only a truncated top-k, which is biased
low exactly where the distribution is flat — the regime the token-level baseline
exists to detect. The in-process binding exposes the whole logit vector, and it
is the only way to teacher-force an answer under a *clean* prompt to get the
unconfounded `h_tok` §4 asks for. (`backends/base.py` still provides
`entropy_bounds_from_topk` if a truncated transport is ever unavoidable; it
returns an interval so an estimate can never masquerade as a measurement.)

On the 2×5090 machine:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install -e ".[llamacpp,nli,data]"

python -m vc_uq all \
  -o model.backend=llamacpp \
  -o model.name=llama-3.1-8b-instruct-q6 \
  -o model.llamacpp.tensor_split="[0.5, 0.5]" \
  -o embedding.backend=llamacpp \
  -o nli.backend=hf
```

Set `model.llamacpp.repo_id` / `filename` (or `model_path`) for the GGUF, and
`embedding.llamacpp.*` for the embedding model. `nli.backend=hf` uses a
DeBERTa-MNLI head; `llm` routes entailment through the same GGUF, which is
weaker and should be reported as such.

## The order things must happen in

`vc_uq all` follows §10, with one deviation. The protocol lists the temperature
sweep first as the cheapest standalone probe; in the full pipeline the Phase 0
gate runs before it, so the sweep uses the validated `tau_star` rather than a
provisional threshold. Run `vc_uq step1` on its own to get the protocol's
literal ordering — it will say loudly that its `tau` is provisional.

`tau` must be held **fixed across the whole sweep**. Re-deriving it at each
temperature lets the correctness threshold move with `T` and confounds the very
effect being measured.

## Phase 0 is a gate, and it needs hand labels

`vc_uq gate` writes a stratified sheet of ~300 `(a_i, a_star)` pairs to
`<run dir>/tables/phase0_label_sheet.csv` and exits. Fill `correct_human` and
re-run with `--labels <path>`. Stratification is across datasets *and* across
the `e_cos` range: a random sample is almost all easy and says nothing about the
boundary where `tau` sits.

The gate is `kappa >= 0.7` AND `AUROC >= 0.85`. Below the AUROC threshold the
primary criterion flips automatically to NLI bidirectional entailment and cosine
is retained as an ablation.

**On the null band.** The check compares *known-correct* pairs against
mismatched `(a_i, a_star_j)` pairs, not all observed pairs. Pooling everything
would understate the instrument, because a wrong answer *is* a mismatched pair
in every respect an embedding can see — a model with a high error rate would
look like a blunt metric. In the smoke run the two numbers are 0.94 and 0.62;
only the first is about the instrument.

`--simulate-labels` substitutes known truth for hand labels. It works only on
the simulated backend, and the gate report records which produced the labels.

## What the simulated backend is for

`backends/mock.py` is not a stand-in for results. It is a world with a hand-set
latent `p_q`, error-correlation structure, and VC miscalibration, so each
estimator can be checked against the value it is supposed to recover.
`tests/test_recovery.py` does exactly that, and its second half is the important
half: it builds a world where VC **is** a good estimator and asserts the analysis
says so. Without that, every negative finding here would be unfalsifiable — code
that always concludes "VC fails" would pass a suite that only tested failing
worlds.

Knobs live under `model.mock` in the config. `vc_reads_answer` is the one that
flips the central prediction: at 0, post-hoc VC ignores the answer it just
produced and within-question AUROC sits at 0.5; at 1 it reads it and the AUROC
goes above 0.9.

## Things the code refuses to do

- **Threshold the same function for membership and correctness.** Every retained
  answer would be correct by construction and the loss would be identically
  zero. `judge.guard_against_circularity` raises.
- **Use `e` as a nonconformity score.** `e` is anchored on `a_star`, so on a new
  question there is nothing to sweep and it cannot be evaluated at all. Only
  `s_anchor` is a valid score.
- **Fixed-width ECE.** On a support of {0.7, 0.8, 0.9, 0.95} it is meaningless;
  `reliability()` raises rather than computing it.
- **Binomial CIs on answer-level reliability.** Every interval resamples
  questions. `test_cluster_bootstrap_is_wider_than_naive_binomial` pins the
  difference at more than 3×.
- **Default a failed VC parse to 0.5.** It returns `None` and the failure rate is
  reported, because unreliable elicitability is itself a finding.
- **Rescale an out-of-range VC.** A model answering `85` when asked for [0,1] is
  a scale-invariance violation and is recorded as one.
- **Decide `U` membership from calibration draws.** `partition_U` reads the
  `classify` split only; using the same draws to classify and calibrate is
  selection on the outcome being certified and voids the LTT guarantee.
- **Use plain Hoeffding.** `hoeffding_bentkus_p` takes the min of both bounds.
  At exactly `R̂ = 0` Hoeffding is the tighter one; the moment `R̂` lifts off zero
  Bentkus binds hard, and that is the realistic case.

## Pitfalls as code

`pitfalls.py` turns §11 into 24 executable checks over the actual artifacts,
written to `results/tables/*__pitfalls.csv`. `fatal` invalidates a result;
`warn` needs a sentence in the writeup. `tests/test_guards.py` constructs the
situation each check exists to catch and asserts it fires — and asserts the
default config passes, so the checks are not unconditionally pessimistic.

## Known limitations

- The TriviaQA loader falls back to a deterministic synthetic stand-in when
  `datasets` is not installed. It labels itself `triviaqa_synthetic` so it can
  never be mistaken for real data in a table.
- LTT needs roughly 500 calibration questions at `alpha = delta = 0.1` for a
  non-vacuous Bentkus bound, and more at `alpha = 0.05`. Below that, nearly
  nothing certifies — visible in the smoke run, where only `alpha = 0.2`
  produces certified configurations.
- `phase4.quality_score: neg_s_anchor` makes retention agreement-with-anchor,
  which biases toward mode collapse. It is an ablation, not the default.
- Per-position token entropies are stored for a 10% subsample only
  (`generation.token_stats.store_per_position_fraction`); summaries are kept for
  every draw.
