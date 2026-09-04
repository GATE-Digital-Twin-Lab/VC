# Verbalised Confidence as an Uncertainty Estimator: Implementation Spec

**Purpose.** Evaluate whether verbalised confidence (VC) functions as an estimator of a
model's per-sample correctness probability. The thesis is *not* "VC is uninformative"
(likely false — see Tian et al., *Just Ask for Calibration*). The thesis is:

> VC violates invariance properties any estimator of `p_q` must satisfy; its errors are
> correlated across samples of the same question, so confidence-based aggregation fails
> and fails *worse* with more evidence; and its dynamic range spans roughly one sample of
> operational difference.

---

## 0. Definitions

| Symbol | Type | Meaning |
|---|---|---|
| `q` | str | A question |
| `a_star` | str | Ground-truth answer |
| `a_i` | str | i-th sampled answer, `i = 1..N_MAX` |
| `T` | float | Sampling temperature |
| `N_MAX` | int | Per-question sampling cap (default 40) |
| `VC_post(q,a_i)` | [0,1] | **Post-hoc** VC — confidence in a specific produced answer. What the literature usually means |
| `VC_pre(q)` | [0,1] | **Pre-hoc** VC — "how confident are you that you can answer this?", elicited before any answer exists. A *different construct*: prospective feeling-of-knowing, not confidence in an output |
| `VC_1(q)` | [0,1] | `VC_post(q, a_1)` — first-draw post-hoc VC. Question-level statistic a single-shot user actually sees |
| `VC_bar(q)` | [0,1] | Mean of `VC_post(q,a_i)` over draws |
| `v_g` | [0,1] | Distinct VC value defining group `g`. **Answer-level** grouping by `VC_post` is primary; question-level grouping by `VC_pre` / `VC_1` is secondary |
| `n_g` | int | Size of group `g` |
| `cluster_i` | int | Semantic cluster id of `a_i` (bidirectional entailment) |
| `H_tok(a_i)` | >=0 | **Mean token entropy** of `a_i`: mean over positions of `-sum_v p_v log p_v` from the full next-token distribution. Family-1 baseline |
| `logp_i` | float | Mean token logprob of `a_i` |
| `minp_i` | [0,1] | Min chosen-token probability over positions of `a_i` |
| `f_i` | [0,1] | Self-consistency: frequency of `a_i`'s cluster among the N samples |
| `H_sem(q)` | >=0 | Semantic entropy over clusters |
| `anchor(q)` | str | The model's own reference answer for `q`. Greedy decode, or the medoid of the N draws. **Deterministic** — see below |
| `s(q, a)` | [0,2] | `1 - cos(emb(anchor(q)), emb(a))`. **Label-free nonconformity score.** `a` is the swept argument and may be anything: `a_star`, a sampled answer, a candidate. Computable at test time |
| `d(a_i, a_j)` | [0,2] | `1 - cos(emb(a_i), emb(a_j))` — pairwise, used for duplicate gating |
| `e(a_i)` | [0,2] | `1 - cos(emb(a_i), emb(a_star))`. **Correctness criterion only.** `a_star` is a fixed reference, not a swept argument, so this is undefined at test time and is used solely to *evaluate* |
| `tau` | float | Admissibility threshold; `correct(a_i) := e(a_i) <= tau` |
| `tau_star` | float | Value selected in Phase 0 |
| `c_i` | {0,1} | `1[correct(a_i)]` |
| `p_q` | [0,1] | **True** per-sample correctness probability. The estimand VC implicitly claims to report |
| `p_hat_q` | [0,1] | `mean(c_i)`. SE <= `0.5/sqrt(N)` |
| `K_q` | int | Draws until first correct answer. **Right-censored at N_MAX** |
| `S(k)` | [0,1] | Kaplan-Meier survival, `P(K_q > k)` |
| `h(k)` | [0,1] | Hazard, `P(c_k=1 | c_1..c_{k-1}=0)` |
| `beta` | [0,1] | KM plateau height = fraction of questions with no admissible answer at any budget |
| `U` | set | `{q : no admissible answer in N_MAX draws}` |
| `A` | set | Complement of `U` |
| `alpha` | float | Target risk. **An input — hitting it validates nothing** |
| `delta` | float | LTT confidence: `P(R(lambda_hat) <= alpha) >= 1 - delta` |
| `lambda` | tuple | CLM config `(lambda_qual, lambda_div, lambda_stop)` |
| `Lambda_k` | (-inf,0] | `sum_{i<=k} log(1 - VC(q,a_i))` over *diverse retained* answers. **The product rule, accumulated in log space** |
| `Pi_k` | [0,1] | `exp(Lambda_k)` — the same claim as a probability. Reporting, and the units of `lambda_stop`; never *accumulated* directly |
| `eps_vc` | float | Floor on `1 - VC` before the log. Default `1e-6` |
| `c_discount` | float | `alpha / exp(lambda_stop_hat)` — the discount factor on VC's claims |

**Critical distinction — `s` and `e` are different functions and must not be merged.**

The test is whether calibration and test time are the *same operation*:

- `s(q, a) = 1 - cos(emb(anchor(q)), emb(a))`. The reference is `anchor(q)`, which the
  model produces. `a` is the argument. At calibration you evaluate at `a = a_star`; at
  test you sweep over candidates. Same operation, no label needed. **Valid score.**
- `e(a_i) = 1 - cos(emb(a_i), emb(a_star))`. Here `a_star` is the *reference* and `a_i` is
  the argument. Given a new question there is nothing to sweep — the function cannot be
  evaluated at all. **Not a score; a correctness criterion.**

Both are needed. `s` builds sets; `e` judges them. Correctness fundamentally requires
`a_star`, so `e` cannot be eliminated — it is confined to evaluation.

**The anchor must be deterministic.** If `anchor(q)` is a fresh sample, the set is random
and coverage marginalises over that randomness too. Use greedy decode (`T=0`), or the
**medoid of the N draws** — the medoid variant is preferable because the radius around it
*is* semantic dispersion, so `s` doubles as the diversity measure in 6.6 rather than
being computed separately. Whichever is chosen, it must be constructed identically at
calibration and test.

Other label-free scores remain available and should be compared: VC, `f`, `H_sem`,
`h_tok_mean`.

**Baseline families.** Comparators must span mechanisms, or a null result cannot be
attributed to VC specifically:

| Family | Signals | Computed from | Blind to |
|---|---|---|---|
| Token-level | `h_tok_mean`, `min_token_p`, `logp_mean` | full next-token distributions, one pass | fluent memorised falsehoods — surface form, not semantics |
| Sample-diversity | `f`, `H_sem` | spread across N draws | mode-collapsed wrong answers (low-diversity `U`) |
| | *(descriptive only — undefined after one draw, so not an online stopping rule; see section 7)* | | |
| Verbalised | `vc_pre`, `vc_post` | generated text | *under test* |

Low token entropy on a wrong answer, and low semantic entropy on a wrong answer, are
**different failures** — the first is stepwise fluency, the second is mode collapse.
Baselines are not included because they are good; they are included to establish whether
VC is worse than signals available for free.

**Circularity guard.** Never define set membership *and* correctness by thresholding the
same function. If `C = {a : s(a) <= lambda}` and `correct(a) := s(a) <= tau`, then every
member of `C` is correct by construction and any loss is identically zero.

---

## 1. Repository layout

```
vc-uq/
  config/
    default.yaml            # all knobs; no magic numbers in code
  src/
    generate.py             # Phase 1: sampling + VC elicitation
    judge.py                # correctness: cosine, NLI, LLM-judge
    cluster.py              # bidirectional-entailment clustering
    survival.py             # Phase 3: KM, hazard, beta, U/A partition
    calibration.py          # Phase 2: reliability, AUROC
    clm.py                  # Phase 4: CLM stopping rules
    ltt.py                  # Hoeffding-Bentkus p-values, fixed-sequence testing
    invariance.py           # Phase 5
    transfer.py             # Phase 6
  data/
    raw/                    # generation output (append-only, never mutate)
    processed/
  results/
    figures/
    tables/
  tests/
```

**Caching is mandatory.** Generation is the expensive step. Key every record by
`(model, dataset, q_id, draw_idx, T, prompt_variant, seed)` and make all downstream
phases pure functions of the cached table. Every phase must be re-runnable without
re-sampling.

---

## 2. Data schemas

### `answers.parquet` — one row per sampled answer

| column | type | notes |
|---|---|---|
| `q_id` | str | |
| `dataset` | str | |
| `split` | str | one of `tau_select`, `classify`, `calib`, `eval` — a property of the QUESTION |
| `draw_set` | str | `classify` or `downstream` — which generation pass produced this draw (6.4) |
| `draw_idx` | int | 0..N_MAX-1, within the pass |
| `answer` | str | |
| `vc_post` | float | `VC_post(q,a_i)`, nullable if parse failed |
| `vc_post_raw` | str | raw model string — keep for parse audit |
| `h_tok_mean` | float | mean token entropy over full vocab distribution |
| `h_tok_max` | float | max per-position token entropy |
| `logp_mean` | float | |
| `min_token_p` | float | |
| `n_tokens` | int | |
| `temperature` | float | |
| `prompt_variant` | str | id of elicitation prompt |
| `seed` | int | |
| `cluster_id` | int | filled by `cluster.py` |
| `e_cos` | float | `1 - cos(emb(a_i), emb(a_star))` — correctness criterion |
| `s_anchor` | float | `1 - cos(emb(anchor(q)), emb(a_i))` — label-free score |
| `correct_cos` | bool | `e_cos <= tau_star` |
| `correct_nli` | bool | bidirectional entailment vs `a_star` |
| `correct_human` | bool | nullable; Phase 0 subset only |

### `questions.parquet` — one row per question

| column | type | notes |
|---|---|---|
| `q_id`, `dataset`, `split` | | |
| `question`, `a_star` | str | |
| `vc_pre` | float | `VC_pre(q)` — pre-hoc, elicited with no answer in context |
| `vc_pre_raw` | str | raw string, parse audit |
| `vc_1` | float | `VC_post(q, a_1)` — first-draw post-hoc VC |
| `vc_bar` | float | mean `vc_post` over draws |
| `vc_sd` | float | sd of `vc_post` over draws — near 0 means VC is a prompt property |
| `p_hat` | float | `mean(correct)` over draws |
| `K_q` | int | first correct draw index, `-1` if censored |
| `censored` | bool | `True` iff no correct answer in N_MAX |
| `in_U` | bool | |
| `n_clusters` | int | |
| `H_sem` | float | |
| `largest_cluster_share` | float | |
| `answer_len_mean` | float | **confounder — must be controlled** |

---

## 3. Phase 0 — Instrument validation (GATE)

Nothing downstream is interpretable if this fails. Run first, on `split == tau_select`.

1. Hand-label ~300 `(a_i, a_star)` pairs. Stratify across datasets and across the range
   of `e_cos` — do not label a random sample, which will be almost all easy.
2. Sweep `tau`; report Cohen's kappa against human labels; select `tau_star`.
3. **Report AUROC of `e_cos` for separating correct/incorrect.**
   - **If AUROC < 0.85, switch to NLI bidirectional entailment as the primary criterion**
     and keep cosine as an ablation. Do not proceed on a blunt instrument.
4. **Null band:** distribution of `e_cos` on mismatched pairs `(a_i, a_star_j)`, `i != j`.
   Run the same null for `s_anchor` using anchors from other questions.
   If the null band overlaps the observed band substantially, the metric cannot resolve
   the effects you are looking for and any flat result downstream is uninformative rather
   than evidence.
5. Every headline result must be re-reported for `tau in [tau_star +/- 0.05]`.

**Known failure mode.** Embeddings are anisotropic — unrelated English sentences sit
around cos 0.3-0.6, and within a fixed domain/format the observed band can be 0.75-0.95
for everything. Negation is invisible: `"Paris"` vs `"not Paris"` scores ~0.9. Consider
mean-centering or whitening embeddings to recover dynamic range; rank-transforming `s`
within a split is also acceptable since only order matters downstream.

**Gate condition:** `kappa >= 0.7` AND AUROC(primary criterion) `>= 0.85`.

---

## 4. Phase 1 — Generation

- `N_MAX = 40` draws per question, fixed `T`, **fresh context per draw** (no chat history,
  otherwise draws are genuinely dependent, not merely modelled as such).
- **Two elicitations, kept strictly separate.**
  - `vc_pre`: "how confident are you that you can answer this question?", **fresh context,
    no answer generated, question shown only.** Elicit `R_pre = 10` times per question to
    get a distribution, not a point — pre-hoc VC has no answer to anchor on, so its
    sampling variance is itself a measurement.
  - `vc_post`: elicited in the same call that produces each answer.
  These are **different constructs**: pre-hoc is prospective feeling-of-knowing (does the
  model know it doesn't know?), post-hoc is confidence in a produced output. Pre-hoc is
  strictly information-poorer — the model has not seen its own answer — which makes the
  comparison `AUROC(vc_pre)` vs `AUROC(vc_1)` a direct measure of **how much information
  seeing its own answer actually adds**. If the two are equal, post-hoc VC is not reading
  the answer at all.
- **Token-level statistics** (local inference — full logits available, so this is free):
  per position, store the entropy of the full next-token distribution. Derive `h_tok_mean`,
  `h_tok_max`, `logp_mean`, `min_token_p`. Storing per-position entropies for all 40 draws
  is large; store summary statistics plus per-position arrays for a 10% subsample.
- Record `n_tokens` per draw (length confounder control).
- **Teacher forcing for token stats.** Compute `h_tok` under the *same* prompt that
  produced the answer. If the VC-elicitation instruction is in the prompt, token entropy
  is measured under that augmented prompt — which is a confound, since VC elicitation
  changes the distribution. Also compute `h_tok` under a clean (non-augmented) prompt by
  teacher-forcing the same answer tokens, and report both.
- **Parse-failure audit.** Log `vc_post_raw`. Report the parse-failure rate; if it is
  non-trivial, note that VC is not reliably *elicitable*, which is itself a finding.
- **Four-way split by `q_id`**: `tau_select` / `classify` / `calib` / `eval`.
  Splits are by question, never by draw.

**Sample size.** `p_hat_q` has SE up to `0.5/sqrt(N)`; at N=10 that is 0.16, comparable to
VC's own granularity. N >= 30 is a floor. For Phase 4, LTT needs `n >= ~500` calibration
questions at `alpha = delta = 0.1` for a non-vacuous Bentkus bound; more at `alpha = 0.05`.

---

## 5. Phase 2 — Descriptive

On `eval`.

1. **VC histograms**, `vc_post` and `vc_pre` separately, with `n_g` per distinct value.
   This quantifies the discreteness claim. Expect support concentrated on 3-5 values in
   [0.7, 1.0]. Report `vc_pre` spread across its `R_pre` repeats per question.
2. **Reliability diagram — answer-level grouping is PRIMARY.** Post-hoc VC attaches to an
   answer, so the claim it makes is: *of all answers assigned 0.8, 80% are correct.* Pool
   all draws across all questions by `vc_post` value and compute `p_hat_g`. VC's
   discreteness supplies canonical bins for free — the model's own partition, not a
   binning heuristic — which sidesteps the arbitrary-bin objection to ECE.
   - **CIs must use a cluster bootstrap resampling QUESTIONS, not answers.** Draws within
     a question are not independent (see 5.5); binomial CIs over 40 x 1000 answers will be
     far too narrow. Effective sample size is closer to the question count.
   - Report calibration gap `|p_hat_g - v_g|` and monotonicity of `p_hat_g` in `v_g`.
   - **Do not use fixed-width ECE.** With values on {0.7, 0.8, 0.9, 0.95} it is
     meaningless. Use adaptive binning or the model-defined-bin version only.
   - Secondary: question-level diagrams grouping by `vc_pre` and by `vc_1`.
3. **VC variance decomposition.** Split `Var(vc_post)` into between-question and
   within-question components. If within-question variance ~ 0, answer-level grouping
   degenerates to question-level and **VC attaches to the prompt regardless of what string
   follows it** — report as a finding, and note it predicts VC-based stopping will tie
   with `fixed_k` in Phase 4.
4. **Pre-hoc vs post-hoc information gain.** Compare `AUROC(vc_pre -> correct)` against
   `AUROC(vc_1 -> correct)` on the same questions. Pre-hoc cannot see the answer, so any
   gap is the information post-hoc VC extracts from reading its own output. **A gap near
   zero means post-hoc VC is not reading the answer either** — it is a question-difficulty
   estimate wearing an answer-confidence label. Report with paired CIs over questions.
5. **Two AUROCs, reported separately** — pooling them conflates distinct claims:
   - **Between-question:** does `vc_pre` / `vc_1` identify hard questions? (target: `p_hat_q`)
   - **Within-question:** for fixed `q`, does `vc_post` rank correct draws above incorrect
     ones? Compute per question, average over questions with both classes present.
     **Near 0.5 means VC is a property of the prompt, not of the answer it ostensibly
     scores** — and it predicts that VC-based stopping will tie with fixed-k in Phase 4.
6. **Brier decomposition**; the *resolution* term is the one that matters.
7. **Within-question correlation** `Corr(c_i, c_j)`, `i != j`. This is the independence
   assumption the product rule needs, stated as one number. Strongly positive ⇒ errors are
   common-mode.

---

## 6. Phase 3 — Survival, U/A partition, diversity (CORE)

### 6.1 Censoring
- **Never average `K_q` over successes only.** That silently deletes exactly the hard
  questions and is the classic bias here.
- Use Kaplan-Meier. Report `S(k)` with CIs. `beta := S(N_MAX)`.
- **Verify the KM curve is flat by `k = N_MAX`.** If still declining, `U` is a censoring
  artifact and everything downstream describes your budget, not the model.
- Report `beta` as a function of `tau` — a strict `tau` inflates it.

### 6.2 Hazard
`h(k) = P(c_k = 1 | c_1..c_{k-1} = 0)`.
- Conditionally-i.i.d. draws with a single `p_q` predict `h(k)` flat in `k`.
- Pooled over heterogeneous `p_q`, `h(k)` must decline — failures are evidence of being in
  a low-`p_q` question.
- Sharp decay to ~0 by k=3 refutes the "just sample more" premise. That is a result.

### 6.3 Predicted vs observed budget
`N_hat_q = ceil(log(alpha) / log(1 - VC_pre(q)))`, and the same with `VC_1`. At
`alpha = 0.1`:

| VC | 0.95 | 0.9 | 0.8 | 0.7 | 0.5 | 0.3 |
|---|---|---|---|---|---|---|
| `N_hat` | 1 | 1 | 2 | 2 | 4 | 7 |

**The observed VC range [0.7, 1.0] maps to `N_hat` in {1, 2}** — VC's entire confidence
vocabulary spans about one sample of operational difference, while true difficulty spans
one sample to never. Report this as the dynamic-range collapse, in units of compute.

### 6.4 U/A partition — ORDER MATTERS
"Unanswerable" is not a label you have; it is *defined by the outcome of generation*. So:

1. **Classification pass.** Draw `N_MAX` on **all** questions. These draws, and only
   these, decide `in_U`, `p_hat`, `K_q`, the KM curve and `beta`.
2. **Downstream pass.** Draw a second, independent `N_MAX` on the `calib`/`eval`
   questions. Phase 2, Phase 4 and 6.7 run on these and never on the first pass.

The held-out thing is the **pass**, not the question. `split` is a property of the
question (4), so "use the `classify` split's draws" would decide `U` for the 30% of
questions in that split using *every draw they have* — the same draws, under another
name — and leave the `calib`/`eval` questions with no membership at all. Both passes
therefore carry a `draw_set` column, and the pass name is salted into the seed so the
second pass is genuinely fresh text rather than a replay of the first under a new label.

Using the same draws to decide membership in `A` and to calibrate is selection on the
outcome being certified and **voids the LTT guarantee**. It also makes the 6.7 `U` curve
vacuous: "none of the first `k` draws was correct" holds *by construction* for every
question the subset was defined to contain, so the observed frequency reads `1.0` at
every `k` whatever VC claimed, and the resulting ratio — 1e16 at `k = 10` in one run — is
arithmetic rather than a finding. Report `n_classify_draws` and `n_downstream_draws`
alongside `beta`.

The cost is one extra generation pass over `calib`/`eval`, about 1.6x total sampling at
the default split shares. There is no cheaper honest version: reserving half of a single
pass for classification would halve the budget that `beta` and the KM plateau are
measured at, which is the one quantity Phase 4 is gated on.

Call `U` what it is: *no admissible answer in `N_MAX` draws under `tau_star`*, not
"unanswerable" in any absolute sense.

### 6.5 The headline comparison
- **AUROC of `vc_pre` for predicting `in_U`**, with CI. Repeat with `vc_1`, `vc_bar`.
  `vc_pre` is the operationally right signal here: deciding to abstain *before* spending
  compute is exactly the pre-hoc question, so if pre-hoc VC detects `U` it has real value.
- Raw overlap, which is more legible than AUROC: e.g. "78% of never-correct questions
  received VC >= 0.8."

**State the hypothesis in the falsifiable direction.** Overlap is the finding; clean
separation is a *positive* result for VC and must be reported as such.

### 6.6 Diversity 2x2
Cluster by bidirectional entailment (**not cosine — negation-blind**). Populate with counts:

| | **A (answerable)** | **U (never correct)** |
|---|---|---|
| **Low diversity** | Genuine knowledge | **Systematic misconception** — one confident wrong belief |
| **High diversity** | Recoverable — correct answer exists but is rare | **Fabrication** — no stable belief |

The low-diversity-`U` cell is the dangerous one: self-consistency, semantic entropy, and
answer frequency are *all* diversity measures, so **none of them can distinguish
confidently-right from confidently-wrong**. If that cell is a meaningful fraction of `U`,
this is a limitation of the field's dominant approach, not just of VC.

Report `AUROC(H_sem -> in_U)` and `AUROC(h_tok_mean -> in_U)` alongside
`AUROC(vc_pre -> in_U)`. All near 0.5 ⇒ no
signal sees the floor.

Also report **VC within each cell**. Expect VC highest on low-diversity-`U`: fluent,
consistent, confident, wrong. If so, VC is anti-correlated with correctness exactly where
stakes are highest.

### 6.7 Product-rule reliability
**Accumulate the product rule as a sum of logs.** The statistic is
`Lambda_k = sum_{i<=k} log(1 - vc_post_i)` over diverse retained answers, never
`prod(1 - vc_post_i)`. Three reasons, only the first of which is cosmetic:

1. The claim decays geometrically. At `vc = 0.99` the linear product underflows float64
   by around `k = 160`, and every question past that becomes an indistinguishable `0.0`.
2. That collapse silently merges the grid quantiles Phase 4 searches over (7.4), so the
   thresholds actually tested depend on where the floating-point floor happens to fall.
3. `vc = 1.0` is **in the observed support** and asserts a failure probability of zero.
   In the linear form one such answer annihilates the product and satisfies every
   threshold at once, so the rule degenerates to "stop as soon as any answer says 1.0."
   Clamp `1 - vc` at `eps_vc` before the log; a single answer's claim is then capped at
   `log(eps_vc)` rather than being unbounded. **`eps_vc` is a declared parameter, not a
   hidden epsilon** — it sets exactly how strong a claim one answer may make, and
   headline results should be re-reported across at least one order of magnitude of it.

Since `log` is strictly increasing, `Pi_k <= t` and `Lambda_k <= log t` cut the sample
space identically: the decision boundary is unchanged and the monotonicity in
`lambda_stop` that licenses fixed-sequence testing (7.4) is preserved. Only the
representation changes.

Bin questions by `Lambda_k`; plot observed frequency of "all k wrong" against
`exp(Lambda_k)`, log-log, one curve per k. **Run separately on `U` and `A`.** Bin on
`Lambda_k` rather than on the exponentiated value — the ordering is the same, but the
linear form produces a wall of exact zeros at large `k` that would collapse the low bins
into one group.

- Honest aggregation ⇒ curves on the diagonal.
- Common-mode errors ⇒ curves flatten, **and the gap widens with k**.

The divergence-in-`k` is the signature: a single VC of 0.8 on a wrong answer is off by ~5x
in failure probability; after five draws the claim is `exp(5 log 0.2) = 3.2e-4` against a
truth of 1, off by ~3000x. Sampling makes the estimate monotonically *worse*. Recalibration
does not repair this — an isotonic map is monotone and pointwise, so it shrinks the values
but preserves the compounding structure, and the gap still diverges in `k`.

**Report the gap as `log10(observed) - Lambda_k/log(10)`, not as a ratio.** In log space the
signature is a straight line in `k` with a positive slope, which is both easier to read and
easier to test than a ratio that reaches `1e5` by `k = 10` and overflows entirely beyond it.
The slope of that line is the quantity to quote: it is the rate at which each additional
sample makes the claim worse. A slope indistinguishable from zero is the falsifying result.

Report the empirical floor where curves plateau: the achievable failure rate regardless of
what the product claims.

---

## 7. Phase 4 — CLM + LTT on `A`

### 7.1 Feasibility check FIRST
`R(lambda) = beta*1 + (1-beta)*R_ans(lambda) >= beta`.

**If `beta > alpha`, then no `lambda` is certifiable, `Lambda_hat` is empty, and the phase
returns a blank table** — not "VC failed," but "no stopping rule of any kind could have
succeeded." Baselines fail identically and the compute is wasted.

- **Primary:** restrict to `A` so `beta_eff = 0` and small `alpha` is feasible. State
  plainly that you conditioned on solvability, which is information unavailable at
  deployment.
- **Robustness:** run on all questions with `alpha > beta` (e.g. `alpha in {0.2, 0.3}`).
- Report `beta` explicitly either way.

### 7.2 The rule
`lambda = (lambda_qual, lambda_div, lambda_stop)`.

- **Retain** `a_i` if quality `>= lambda_qual` AND `min_j d(a_i, a_j) >= lambda_div` over
  already-retained `a_j` (duplicate gating). Diversity gating is not optional: without it
  a mode-collapsed model fills `C(q)` with forty paraphrases of one wrong answer and set
  size stops meaning anything.
- **Stop** when the confidence statistic crosses `lambda_stop`. This is the swappable slot.

**Quality is `logp_mean`, the length-normalised log-likelihood — not VC.** This is what
conformal language modelling admits on, and here it is load-bearing rather than
conventional. Phase 4 exists to test VC as a *stopping* signal against baselines at equal
risk. If VC also decided which answers were retained, then `token_entropy` and even
`fixed_k` would be scored on a set VC had already filtered: no VC-free arm would remain,
and a difference in `E[draws]` could not be attributed to the stopping rule. VC
appears in exactly one place, which is the claim under test. `-s(q, a_i)` is available as
an ablation — it makes retention agreement-with-anchor, which biases toward mode collapse
— and `vc_post` is diagnostic only, with a fatal pitfall check when it is combined with VC
stopping rules.

**Draws with a missing statistic are dropped, never imputed.** An unparsed VC is not a
confidence of `0.0`. Filling one in puts a number the model never produced into the
statistic the study is about, and it is not even neutral: `0.0` reads as "maximally
unsure", so every VC rule draws more and `E[draws]` — the one quantity the headline table
compares — is inflated in proportion to the parse-failure rate. Dropping costs the
baselines that draw too, which is the price of every rule seeing one identical sequence.
Report the dropped fraction: `draws` counts positions in the retained sequence, so
`E[draws]` understates sampling actually spent by about that much.

**Every component of `lambda` is searched over `[0, 1]`, except `vc_product`'s `lambda_stop`.** Each underlying score is
rescaled to the unit interval first, so one search space serves all three dimensions and
all seven rules, and a certified threshold is readable without a units table:

| component | score | scaling |
|---|---|---|
| `lambda_qual` | `p(y\|x)^(1/T) = exp(mean_t log p_t)` | already `[0,1]` — a probability |
| `lambda_div` | cosine distance | `d / 2`, since `d` spans `[0, 2]` |
| `lambda_stop` (`vc_max`) | probability | already `[0,1]` |
| `lambda_stop` (`vc_product`) | `Lambda_k`, a log-probability | **not rescaled** — kept in nats; `-inf` means "never stop early" |
| `lambda_stop` (`token_entropy`) | `exp(-H)`, the geometric mean token probability | `H` is unbounded above; `exp(-H)` is in `(0,1]` and preserves the ordering |
| `lambda_stop` (`fixed_k`) | `\|C(q)\| / N_MAX` | fraction of the budget |

Divide the cosine distance by 2 rather than clipping at 1: clipping would collapse every
anti-correlated pair, 18% of them on the corpus this was measured on.

**`vc_product` stays in nats on both sides, and that exception is the point.** Its claim
compounds, so the thresholds that matter span orders of magnitude: as probabilities they
bunch against 0 and stop being readable, while `-5` and `-20` nats are two legible
numbers — on the same axis 6.7 reports the product-rule gap on. Exponentiating `Lambda_k`
to meet a `[0, 1]` threshold would also reproduce the wall of exact zeros 6.7 warns about.
Grid it evenly in nats from `log(product_floor)` to `0` (which is geometric in the claim
it represents), plus `-inf` for "never stop early" — the only threshold guaranteed to stay
reachable at any `N_MAX`, and the conservative end the fixed sequence starts from.

Because `lambda_stop_hat` is a log-probability, `c_discount = alpha / exp(lambda_stop_hat)`
— exponentiate before dividing. Dividing `alpha` by a threshold in nats is a units error
that still produces a plausible-looking number.

Every remaining map is monotone increasing in `lambda_stop`, which is what lets the
certification order (7.4) be read off the rule's direction in `lambda` space.

Watch the count of *distinct* `(lambda_qual, lambda_div)` cells. `delta` is divided by the
number of cells (7.4), so grid points that behave identically are lost power, not merely
wasted compute. The failure to avoid is a grid placed where the scores are not: on
`vc_post`, whose observed range is `[0.7, 1.0]`, seven of ten evenly spaced `lambda_qual`
points were the *identical test*, and only 8 of 100 cells were genuinely distinct.

**Every rule reads `C(q)`, never the raw draw stream.** The gate decides what is in the
set; the rule then reads the set. An answer the quality or diversity gate rejected is not
in `C(q)`, so it must not be the evidence that ends sampling — stopping because of an
answer you then throw away would certify a set that never contained it.

This is also what makes `lambda_qual` and `lambda_div` mean anything. While the statistics
ran over all draws, gating changed the set but never the draw count: retention could not
affect efficiency, so it was never selected at any `alpha`, and `E[|C(q)|]` was identically
equal to `E[draws]`. Reading `C(q)` restores the trade the two dimensions exist to offer —
a stricter gate admits less, the statistic advances more slowly, the rule spends more
draws, and in exchange the low-quality answers that would have triggered a premature stop
are gone.

| Variant | Stop when | Family |
|---|---|---|
| `vc_product` | `Lambda_k = sum log(1 - vc_post_i) <= lambda_stop` over `C(q)`. **`lambda_stop` is in log-probability (nats) for this rule** | verbalised |
| `vc_max` | the most confident answer **in `C(q)`** reaches `lambda_stop` | verbalised |
| `token_entropy` | the lowest-entropy answer **in `C(q)`** reaches `lambda_stop`, as `exp(-H)` | token-level |
| **`fixed_k`** | **`\|C(q)\| >= lambda_stop * N_MAX`** — **the null baseline** | none |

`fixed_k` counts the *set*, not draws. Counting draws would let the null ignore the gate
every other rule pays for, so it would be playing a different game rather than marking the
floor.

Three rules were retired. `vc_first` and `vc_prehoc` fix the budget from a single number —
the first draw's VC, or a pre-hoc estimate — so they cannot read `C(q)` at all and belong
to a different family from everything here; the marginal-value question they answered is
better put to Phase 2 and 6.5, where `vc_pre` and `vc_1` are already compared against
`vc_bar` on the same questions. `min_token_p` duplicated the token-level mechanism
`token_entropy` already carries, and a second near-identical baseline costs multiplicity
(7.4) without adding an argument.

The sample-diversity signals are **not** stopping rules here. A diversity
statistic over one draw is not a low value, it is not a value: one draw is one
cluster, so `H_sem = 0` and largest-share `= 1` — exactly the values a
"stop when converged" rule reads as convergence. Both would therefore fire at
`k = 1` for every threshold, making them duplicates of `fixed_k = 1` under names
that imply otherwise. The alternative, a minimum draw count before either may
fire, changes the efficiency each rule reports and so alters the very quantity
the table compares. They are reported descriptively instead (section 6.6 and the
Phase 2 AUROC comparison), which is where a statistic over a completed set of
`N` draws belongs.

`fixed_k` is mandatory. If VC-based stopping cannot beat "always draw exactly k," VC
carries no usable information about when to stop, and that is the cleanest statement of it.

The marginal value of each extra piece of evidence VC conditions on — nothing, one
answer, all of them — is measured in Phase 2 and 6.5 instead, where `vc_pre`, `vc_1` and
`vc_bar` are compared against the same outcome on the same questions. That comparison does
not require any of them to be a stopping rule, and a rule that cannot read `C(q)` has no
place in a table whose whole point is the price of a set.

### 7.3 Risk
`L_lambda(q) = 1[no admissible answer in C_lambda(q)]`, `R(lambda) = E_q[L_lambda(q)]`.

Bounded in [0,1] and monotone in `lambda_stop` (stopping later can only help) — this
monotonicity is what licenses fixed-sequence testing below. `R` inherits the judge's error
rate; report at multiple `tau`.

### 7.4 LTT
Split conformal does not apply here: the risk is set-valued ("contains at least one
admissible answer"), the set is built adaptively, and `lambda` is multidimensional. There
is no single score whose quantile solves it.

1. **Grid** `Lambda` over the three components. Keep it coarse (10x10x10) — power is lost
   to multiplicity.
2. **p-value** per `lambda`, Hoeffding-Bentkus:
   ```
   p_lambda = min(
       exp(-n * h1(min(R_hat, alpha), alpha)),
       e * BinomCDF(ceil(n * R_hat); n, alpha)
   )
   ```
   where `h1(a,b) = a*log(a/b) + (1-a)*log((1-a)/(1-b))`.
   **The Bentkus term dominates when `R_hat` is near 0** — that is your regime at
   `alpha = 0.05`, so do not use plain Hoeffding.
3. **FWER correction.** Bonferroni over the whole grid is valid but wastes power. Use
   **fixed-sequence testing along `lambda_stop`**, one sequence per
   `(lambda_qual, lambda_div)` cell: order from most conservative (stops latest) to
   least, halt at the first non-rejection. Within a sequence there is no multiplicity
   penalty, and that is valid precisely because `R` is monotone in `lambda_stop`.

   Two things this gets wrong if left implicit, neither of which raises:

   - **Which end is conservative depends on the rule's comparison direction.** A `ge`
     rule (`vc_max`, `token_entropy`, `fixed_k`) stops when its statistic *rises* to
     `lambda_stop`, so the strictest setting is the highest and the sequence walks
     **down**. A `le` rule (`vc_product`, `token_entropy`) stops when its statistic
     *falls* to `lambda_stop`, so the strictest is the lowest and the sequence walks
     **up**. Reversed, the sequence opens on its worst grid point, halts immediately and
     returns an empty `Lambda_hat` — which reads as "this rule cannot be certified".
     For `vc_product` that is the study's headline result arriving as a sort order.
     Take the direction from the rule, **never from the observed risks**: a
     data-dependent order is what fixed-sequence validity forbids.
   - **`G` parallel sequences at `delta` each control FWER at `G * delta`.** The grid has
     `G = 10 x 10 = 100` cells. Bonferroni **across** families, fixed sequence **within**:
     each runs at `delta / G`. The correction falls on `delta` alone, so the `lambda_stop`
     resolution stays free. Report `delta_family` next to `delta`.

   Report the count of families whose realised `R` is not monotone along the chosen
   order. Within a family the retention path is fixed, so a stricter threshold can only
   stop later and stopping later can only turn a loss of 1 into a 0 — `R` is exactly
   monotone *in the sample*, and any violation is a defect, not noise.
4. **Output** `Lambda_hat = {lambda : p_lambda < delta / G}`. Guarantee:
   `P(R(lambda) <= alpha for all lambda in Lambda_hat) >= 1 - delta`, which is what
   licenses selecting from `Lambda_hat` afterwards without paying again.

### 7.5 Selection and the headline table
Select `lambda_hat = argmin E_q[draws]` over `Lambda_hat`, **on `eval`, never on `calib`**.

For `alpha in {0.05, 0.1, 0.2}`, `delta = 0.1`:

| stop score | `E[draws]` | `E[|C(q)|]` | realized risk | `lambda_stop_hat` | units | `c_discount` |
|---|---|---|---|---|---|---|
| vc_product | | | | | | |
| vc_max | | | | | | |

| token_entropy | | | | | | |
| fixed_k | | | | | | |

**Risk is held constant by construction; efficiency is the free variable.** That is what
makes the comparison meaningful — same guarantee, different price.

Three outcomes, all publishable:
- VC ~ `fixed_k` ⇒ VC carries no information about when to stop.
- VC worse than token-entropy or diversity signals ⇒ dominated by signals that require no
  prompt augmentation. This is the "prompt must be augmented to elicit VC" objection,
  priced in compute.
- VC competitive ⇒ report honestly; Phase 5 still carries the thesis.

### 7.6 The discount factor
`c_discount = alpha / exp(lambda_stop_hat)` for `vc_product`. **`lambda_stop_hat` is a
log-probability, so it must be exponentiated before dividing** — `alpha` divided by a
threshold in nats is a units error that still yields a plausible-looking number. Form the
quotient in log space and exponentiate once: `log c_discount = log(alpha) - lambda_stop_hat`.
Quote `log10(c_discount)` as the primary figure; the linear form overflows when the
certified threshold sits far below `alpha`, which is precisely the regime being measured.

If VC were truthful, `exp(lambda_stop_hat) = alpha` and `c_discount = 1`. Empirically
expect `>> 1`: the model's
self-reported failure probability must reach `alpha / c` before 1-alpha coverage is
actually achieved.

**Stability of `c_discount` is the load-bearing check.** Recompute per dataset, per
`p_hat_q` stratum, per domain, per elicitation prompt:
- roughly constant ⇒ fixed miscalibration, fixable, modest result;
- varies by an order of magnitude ⇒ no fixed correction works, VC is not a stable
  instrument. **This is the thesis, and it is immune to the recalibration rebuttal.**

### 7.7 Vacuity check
If `Lambda_hat` contains only configurations that draw all `N_MAX` samples and retain
everything, every method ties at the ceiling and the table says nothing. Assert
`Lambda_hat` is non-empty at `lambda_stop` values that stop early; flag loudly otherwise.

### 7.8 Stratify `A`
The interesting subset is not the easy questions but those with low but nonzero `p_hat_q`,
where a correct answer exists but is rare — that is where a stopping rule earns its keep.
If VC is flat across `p_hat_q in (0, 0.3]` while required draws vary by an order of
magnitude, that is the dynamic-range collapse shown where it costs real compute.

---

## 8. Phase 5 — Invariance (carries the thesis)

Phases 2-4 establish poor *quality*, which isotonic regression could in principle fix.
These show VC violates properties any estimator of `p_q` must satisfy.

1. **Temperature sweep**, `T in [0, 1.5]`, prompt fixed. Measure `p_hat_q` (moves a lot)
   and VC (has no argument for `T` and will barely move).
   **`T` must be the only thing that varies.** Truncation sampling (`top_p`, `top_k`,
   `min_p`) and repetition penalties reshape the distribution independently of `T`,
   so they must be disabled and passed explicitly rather than left to backend
   defaults. A truncated tail damps the movement in `p_hat_q` and understates the
   very effect being measured. `tau` must likewise be held fixed across the sweep. `p_q` is definitionally a
   function of the decoder; VC is not. **Cheapest experiment with the strongest payoff —
   run this first.**
2. **Elicitation paraphrase**, 10 semantically equivalent prompts. Report per-question
   variance. This is test-retest reliability, a basic psychometric requirement.
3. **Unanswerable / false-premise questions** (FalseQA, fabricated entities) where
   `p_q ~ 0` by construction — an independent handle on the `U` analysis.

Deliberately **out of scope**: scale reframing, the sycophancy probe, the forced-decode
intervention, and the pre-hoc metacognition probes. Each asked a different question from
the two above — whether VC responds to reporting format, to conversational pressure, or to
its own injected value — and each needed its own control arm to be interpretable. The
invariance claim rests on 8.1 and 8.2, which are matched comparisons: the temperature
sweep holds the prompt fixed and moves only `T`; the paraphrase set holds the question
fixed and moves only wording.

---

## 9. Phase 6 — Recalibration transfer

Fit isotonic regression on dataset A, apply to dataset B, report calibration gap before and
after. Failure to transfer ⇒ not a stable instrument. This pre-empts the obvious rebuttal
that miscalibration is trivially fixable.

---

## 10. Build order

| Step | Deliverable | Gate |
|---|---|---|
| 1 | Phase 5.1 temperature sweep on ~100 questions | Does the headline effect exist at all? Cheapest signal |
| 2 | Phase 0 instrument validation | `kappa >= 0.7`, AUROC `>= 0.85` |
| 3 | Phase 1 full generation + cache | KM flat by `N_MAX` |
| 4 | Phase 3 survival, `beta`, U/A, 2x2 | `beta` known before any `alpha` is chosen |
| 5 | Phase 2 descriptive | |
| 6 | Phase 4 CLM+LTT | `Lambda_hat` non-empty and non-vacuous |
| 7 | Phases 5.2-5.6, Phase 6 | |

**MVP paper:** steps 1-4 plus the temperature sweep. The censoring analysis and decoder
non-invariance are already a contribution.

---

## 11. Pitfalls — check every one before reporting

- [ ] Set membership and correctness defined by the **same** function (tautology; loss ≡ 0)
- [ ] `e` (a_star-anchored) used as a nonconformity score — it is undefined at test time;
      only `s` (anchor-anchored) is a valid score
- [ ] `anchor(q)` drawn stochastically, or constructed differently at calibration vs test
- [ ] `K_q` averaged over successes only (deletes hard questions)
- [ ] KM curve still declining at `N_MAX` (censoring artifact reported as model property)
- [ ] `alpha` chosen below `beta` (empty `Lambda_hat`, blank table)
- [ ] Same draws used to classify `A` and to calibrate (voids guarantee, and pins every
      `U` bin of the 6.7 curve at an observed frequency of 1.0)
- [ ] `in_U` imputed from a question's own censoring where the partition did not cover it
      — the fallback *is* the circularity, and a missing classification draw must stop the
      run instead
- [ ] `tau` selected and LTT run on the same split (voids guarantee)
- [ ] `lambda_hat` selected on `calib` rather than `eval`
- [ ] Sampling truncation left at the backend's defaults, so the decoder is not a
      function of `T` alone. llama.cpp ships `top_k=40, top_p=0.95, min_p=0.05,
      repeat_penalty=1.1`; passing nothing silently truncates the tail, damps the
      effect of raising `T`, and understates 8.1. Pass every knob explicitly.
- [ ] Product rule accumulated as a running product rather than `sum log(1 - vc)`
      (underflows at large `k`; collapses the Phase 4 grid quantiles)
- [ ] `vc = 1.0` left unclamped, so one answer zeroes `Pi_k` and satisfies every threshold
- [ ] `c_discount` computed as `alpha / lambda_stop_hat` without exponentiating the
      log-space threshold (units error; the number still looks plausible)
- [ ] Plain Hoeffding used where `R_hat ~ 0` (Bentkus needed)
- [ ] Fixed sequence ordered by `lambda_stop` alone, ignoring the rule's comparison
      direction, so every `le` rule opens on its worst grid point and reports as
      uncertifiable
- [ ] One fixed sequence per grid cell, each run at the full `delta` (FWER is `G*delta`)
- [ ] Sequence order derived from the observed risks rather than pre-specified
- [ ] VC used as the retention gate as well as the stopping rule, leaving no VC-free arm
      to compare against
- [ ] Unparsed `vc_post` imputed as `0.0` rather than dropped — reads as "maximally
      unsure", so every VC rule draws more and `E[draws]` is inflated
- [ ] A `lambda` component searched on a scale that is not `[0, 1]`, so thresholds are
      not comparable across rules and the grid cannot be read without a units table
- [ ] Cosine distance clipped at 1 instead of divided by 2 (collapses anti-correlated pairs)
- [ ] `Lambda_k` exponentiated to put `vc_product`'s `lambda_stop` on `[0, 1]`
      (reproduces the underflow wall of 6.7, and makes the threshold unreadable)
- [ ] `vc_product`'s `lambda_stop` grid spaced linearly in the CLAIM rather than evenly in
      nats, so the thresholds that matter are never tested
- [ ] Grid points that behave identically, each costing a family's worth of `delta`
- [ ] A stopping rule reading the raw draw stream rather than `C(q)`, so an answer the
      gate rejected still ends sampling -- and `lambda_qual`/`lambda_div` cost
      multiplicity while being unable to change `E[draws]`, so they are never selected
- [ ] `fixed_k` counting draws rather than the size of `C(q)`, so the null baseline skips
      the gate every other rule pays for
- [ ] No FWER correction over the `lambda` grid
- [ ] Cosine used for clustering (negation-blind)
- [ ] Fixed-width ECE on discrete VC values
- [ ] `fixed_k` baseline omitted
- [ ] Binomial CIs on answer-level reliability (draws within a question are correlated —
      **cluster bootstrap over questions**)
- [ ] `vc_pre` elicited with any answer in context (contaminates the construct)
- [ ] `h_tok` computed only under the VC-augmented prompt (confounded — also teacher-force
      under a clean prompt)
- [ ] Pre-hoc and post-hoc VC pooled or treated as the same quantity
- [ ] Answer length uncontrolled (confounds `s` with question type)
- [ ] `n_g` unreported (high-VC groups will dominate; low-VC groups may be tiny)
- [ ] Between- and within-question AUROC pooled
- [ ] Conclusion flips under `tau_star +/- 0.05`

---

## 12. Claims to defend, in order of strength

1. VC's errors are **correlated across samples of the same question**, so confidence-based
   aggregation fails and the gap **widens with `k`**. Not repairable by any monotone
   recalibration.
2. VC fails **invariance** properties any estimator of `p_q` must satisfy: not a function
   of the decoder, not stable under elicitation paraphrase or reporting scale, and
   perturbs the distribution it claims to measure.
3. VC's **dynamic range** spans ~1 sample of operational difference across its entire
   observed support.
4. VC is **blind to the infeasible floor** `U`, where no sampling budget succeeds and
   abstention is the only correct action.
5. Diversity-based signals (self-consistency, semantic entropy) **also** fail on
   low-diversity-`U`, so this is a limitation of the dominant approach, not only of VC.
   This claim rests on the 6.6 diversity 2x2 and the Phase 2 AUROC comparison, where
   `f` and `H_sem` describe a completed set of `N` draws — not on the stopping table,
   which they cannot enter for the reason given in section 7. The stopping table still
   spans two mechanisms, token-level (`token_entropy`) and verbalised (`vc_product`,
   `vc_max`), against a null that uses neither, so "VC fails here" can be separated from
   "everything fails here."
6. **Pre-hoc VC ("can you answer this?") is a distinct and under-studied construct.**
   The ladder `vc_pre -> vc_1 -> vc_bar` measures the marginal value of each additional
   piece of evidence VC conditions on, in Phase 2 and 6.5 rather than as stopping rules.
   If flat, post-hoc VC is not reading its own answer and the whole post-hoc framing is
   mislabelled question-difficulty estimation.

Do **not** claim VC is uninformative. Tian et al. found verbalised confidence from RLHF'd
models often beats conditional token probabilities on calibration; that result will be
raised, and the claims above survive it. It also cuts the other way here: since that
comparison is the standard one in the literature, excluding token probability from the
baseline table needs an explicit justification in the writeup.

---

## 13. References

- Angelopoulos, Bates, Candès, Jordan, Lei — *Learn Then Test: Calibrating Predictive
  Algorithms to Achieve Risk Control*
- Quach, Fisch, Schuster, Yala, Sohn, Jaakkola, Barzilay — *Conformal Language Modeling*
- Mohri, Hashimoto — *Language Models with Conformal Factuality Guarantees*
- Kuhn, Gal, Farquhar — *Semantic Uncertainty*
- Farquhar, Kossen, Kuhn, Gal — *Detecting Hallucinations Using Semantic Entropy*
- Tian, Mitchell, Zhou, Sharma, Rafailov, Yao, Finn, Manning — *Just Ask for Calibration*
- Barber, Candès, Ramdas, Tibshirani — *The Limits of Distribution-Free Conditional
  Predictive Inference*
- Vovk — *Conditional Validity of Inductive Conformal Predictors*
- Ethayarajh — *How Contextual are Contextualized Word Representations?* (anisotropy)
- Gao et al. — *Representation Degeneration Problem in Training NLMs*