# Verbalised Confidence (VC) — Annotated Bibliography

**Project thesis.** Verbalised confidence is not a principled uncertainty-quantification signal. It has no
estimator-level definition, its support is a handful of round-number atoms, it is determined by the
elicitation prompt and the preceding token context, and the act of eliciting it perturbs the model's
behaviour.

**Status of entries.** Every arXiv ID, title, author list, and date below was checked against the arXiv
abstract page or a search result on 2026-08-24. Entries marked ⚠️ carry a caveat about how the paper was
described in our working notes vs. what it actually says — read those before citing.

**Claim numbering used throughout:**

| # | Claim |
|---|-------|
| C1 | No formal grounding — VC is not an estimator of any well-defined quantity |
| C2 | Discreteness — VC takes ~3–5 values, clustered high; breaks ECE |
| C3 | Prompt- and context-dependence — VC is a function of the elicitation, not the belief |
| C4 | The elicitation prompt is itself harmful — asking changes the model |

---

## 1. The canon we argue against (pro-VC lineage)

### Teaching Models to Express Their Uncertainty in Words
- **Authors:** Stephanie Lin, Jacob Hilton, Owain Evans
- **arXiv:** [2205.14334](https://arxiv.org/abs/2205.14334) — 28 May 2022 (rev. 13 Jun 2022) · TMLR
- **Role:** Origin point of the VC literature. Trains GPT-3 to emit "90% confidence" / "high confidence"
  alongside answers; introduces the CalibratedMath benchmark. Claims the first demonstration of a model
  expressing *calibrated* uncertainty in words, with partial robustness under distribution shift.
- **Use:** Establishes the construct. Note the scope: a single fine-tuned model, one synthetic arithmetic
  domain, calibration measured in-distribution. Nothing in it defines what VC estimates.

### Language Models (Mostly) Know What They Know
- **Authors:** Saurav Kadavath, Tom Conerly, Amanda Askell, et al. (Anthropic)
- **arXiv:** [2207.05221](https://arxiv.org/abs/2207.05221) — 11 Jul 2022 (rev. 21 Nov 2022)
- **Role:** Introduces P(True) (model scores its own answer) and P(IK) (model predicts whether it knows).
  Larger models are well calibrated on multiple choice.
- **Use:** Important distinction for C1 — P(True) *is* a token probability of a well-defined proposition,
  so it has an estimator-level reading that free-form VC lacks. Don't lump it in with VC. It fails on
  robustness grounds (see 2601.08064) rather than definitional grounds.

### Reducing Conversational Agents' Overconfidence through Linguistic Calibration
- **Authors:** Sabrina J. Mielke, Arthur Szlam, Emily Dinan, Y-Lan Boureau
- **arXiv:** [2012.14983](https://arxiv.org/abs/2012.14983) · **TACL** 10 (2022) 857–872 —
  [ACL Anthology](https://aclanthology.org/2022.tacl-1.50/)
- **Role:** Linguistic (verbal-hedge) calibration for chit-chat models. Finds models poorly calibrated,
  but that correctness *is* predictable from internal signals — then trains the model to hedge accordingly.
- **Use:** Precursor to the 2607.08046 result — the signal exists internally; the verbalisation has to be
  *forced* to track it. Also the natural bridge to the human psychometric literature (§6).

### Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback
- **Authors:** Katherine Tian, Eric Mitchell, Allan Zhou, et al. (Stanford)
- **arXiv:** [2305.14975](https://arxiv.org/abs/2305.14975) — May 2023 · **EMNLP 2023**
- **Role:** ⚠️ **The paper to engage most directly.** Claims RLHF-tuned models' verbalised confidences are
  better calibrated than their own conditional token probabilities — ~50% relative ECE reduction on
  TriviaQA, SciQ, TruthfulQA.
- **Rebuttal line:** (i) the comparison is protocol-dependent — 2605.27752 shows the conditioning context
  alone flips the *sign* of the ECE gap; (ii) the baseline is a *known-broken* RLHF logit, not a good UQ
  method — beating a miscalibrated baseline is not evidence of a valid estimator; (iii) low ECE under a
  ~5-atom support is cheap (our C2 formal result).

### Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
- **Authors:** Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, Bryan Hooi
- **arXiv:** [2306.13063](https://arxiv.org/abs/2306.13063) — 22 Jun 2023 (rev. Mar 2024) · **ICLR 2024**
- **Role:** The standard black-box VC benchmark (prompt / sample / aggregate framework, 5 LLMs).
- **Key numbers:** White-box beats black-box, AUROC gap 0.522→0.605. Explicitly reports that LLMs
  verbalising confidence "tend to be overconfident, potentially imitating human patterns."
- **Use:** Supports C2 from *inside* the pro-VC camp, and the "imitating human patterns" line is our
  hand-off into the psychometric literature (§6).

### SteerConf: Steering LLMs for Confidence Elicitation
- **Authors:** Ziang Zhou et al.
- **arXiv:** [2503.02863](https://arxiv.org/abs/2503.02863) — 4 Mar 2025 · **NeurIPS 2025**
- **Role:** Source of the standard motivating framing: confidence elicited by prompting, for black-box API
  models where internal scores are inaccessible. Steers with conservative/optimistic prompts, aggregates
  by consistency; +29% AUROC, −16% ECE from a "very cautious" prompt.
- **Use:** Double-edged and useful. The *accessibility* argument is why VC persists (cheap, model-agnostic
  — not principled). And the result itself is direct C3 evidence: if a tone adjective moves AUROC 29
  points, the score is a function of the prompt.

---

## 2. C2 — Discreteness / saturation

### Rescaling Confidence: What Scale Design Reveals About LLM Metacognition
- **Authors:** Yuyang Dai, Yuxia Wang
- **arXiv:** [2603.09309](https://arxiv.org/abs/2603.09309) — Mar 2026
- **Role:** ⭐ **Closest published statement of our C2.** Coins "confidence discretization."
- **Key numbers:** >78% of responses fall on just **three** round-number values, across 6 LLMs × 3 datasets.
  Round-number preference persists under *irregular* ranges. 0–20 scale beats 0–100 on metacognitive
  efficiency (meta-d′); boundary compression degrades it.
- **Their mechanism claim (verbatim thesis match):** VC is produced by token-level selection, not by a
  continuous internal estimate mapped onto a scale; under 0–100, high-frequency round tokens (90, 95, 100)
  act as attractors. They note this potentially distorts ECE.
- **Differentiation:** They stop at "scale design guidelines." We take the same finding to the formal
  conclusion — under ~5 atoms of support, a constant predictor achieves low ECE and the metric is
  uninformative. That is the publishable gap.

### Calibrating Verbalized Confidence with Self-Generated Distractors (DINCO)
- **Authors:** Victor Wang, Elias Stengel-Eskin
- **arXiv:** [2509.25532](https://arxiv.org/abs/2509.25532) — 29 Sep 2025 ·
  [OpenReview](https://openreview.net/forum?id=pZs4hhemXc)
- **Role:** Names **"confidence saturation"** — scores taking few unique values, clustered high. Attributes
  overconfidence to *suggestibility bias*: the model endorses whatever claim it is shown when it encodes
  little about it. Normalises verbalised confidence over self-generated distractors.
- **Key numbers:** DINCO at 10 inference calls > self-consistency at 100.
- **Use:** Independent naming of the same pathology, plus a mechanism (suggestibility) that is itself a C3
  argument — the score tracks the presented claim, not the model's state. ⚠️ It is a *fix* paper; cite for
  the diagnosis, note that the fix concedes the diagnosis.

### Verbal Confidence Saturation in 3–9B Open-Weight Instruction-Tuned LLMs: A Pre-Registered Psychometric Validity Screen
- **Authors:** Jon-Paul Cacioli
- **arXiv:** [2604.22215](https://arxiv.org/abs/2604.22215) — 24 Apr 2026
- **Role:** Pre-registered validity screen — the most methodologically clean C2 evidence.
- **Key numbers:** All 7 models fail validity under numeric (0–100) elicitation; **mean ceiling rate 91.7%**.
  Categorical (10-class) elicitation does not help and drops accuracy below 5% in 6/7 models. Logprob
  signals predict verbalised confidence at **cross-validated R² < 0.01**.
- **Use:** That R² is a strong faithfulness result for C1/C4 — the verbalised number is near-unrelated to
  the model's own token-level signal. Small models only; scope accordingly.

---

## 3. C3 — Prompt, protocol, and adversarial sensitivity

### On Verbalized Confidence Scores for LLMs
- **Authors:** Daniel Yang, Yao-Hung Hubert Tsai, Makoto Yamada
- **arXiv:** [2412.14737](https://arxiv.org/abs/2412.14737) — 19 Dec 2024 (rev. 5 May 2026)
- **Role:** ⚠️ **Both a key C3 reference and a live counterargument.** Documents that the literature openly
  disagrees about VC calibration and attributes the disagreement to prompt method — reliability depends
  strongly on how the model is asked. *But* concludes it is possible to extract well-calibrated scores
  with certain prompt methods, and advocates VC as a low-overhead, model-agnostic UQ method.
- **Rebuttal line:** prompt-conditional calibration is not calibration. If it holds only for a prompt found
  by search over prompt space, that is overfitting to the eval set, not a property of an estimator.

### Calibration Is Not Enough: Evaluating Confidence Estimation Under Language Variations
- **Authors:** Yuxi Xia, Dennis Ulmer, Terra Blevins, Yihong Liu, Hinrich Schütze, Benjamin Roth
- **arXiv:** [2601.08064](https://arxiv.org/abs/2601.08064) — 12 Jan 2026
- **Role:** ⭐ Ready-made evaluation framework — exactly the necessary-conditions structure our paper needs.
  Three properties: (1) robustness to prompt perturbation, (2) stability across semantically equivalent
  answers, (3) sensitivity to semantically different answers. Tested across scale variants (0–1, 0–100%,
  0–10), lexical variants ("probability"/"certainty"/"confidence"), and verbal scales.
- **Key findings:** These metrics are largely *independent* of existing CE metrics — i.e. low ECE tells you
  nothing about them. Most methods are robust/stable but fail sensitivity. P(True) and VC are the least
  robust under prompt perturbation.
- **Use:** Cite as the empirical instantiation of our "invariance to semantically-null perturbation"
  necessary condition. The independence-from-ECE result is central.

### Asking Is Not Enough: Protocol Sensitivity in LLM Confidence Calibration
- **Authors:** Hankyeol Kim, Pilsung Kang (Seoul National University)
- **arXiv:** [2605.27752](https://arxiv.org/abs/2605.27752) — Jun 2026
- **Role:** ⭐ The methodological version of our argument, and the direct answer to Tian et al. Varies which
  answer string is scored, how the token probability is read, and the conditioning context.
- **Key finding:** **Conditioning context changes the sign or magnitude of the ECE gap** between VC and
  token probability; token readout moves the sign too; the ECE estimator barely matters. Concludes both
  signals are *protocol-dependent behavioural measurements* and gives a reporting checklist.
- **Use:** This is the load-bearing citation against "VC beats token probabilities." The claim is not
  merely unreplicated — it is under-specified.

### On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks
- **Authors:** Stephen Obadinma, Xiaodan Zhu
- **arXiv:** [2507.06489](https://arxiv.org/abs/2507.06489) — 9 Jul 2025 (rev. 18 Dec 2025) · **NeurIPS 2025**
- **Role:** Adversarial ceiling on C3. First comprehensive robustness study; four perturbation-based and two
  jailbreak-based attacks.
- **Key findings:** Attacks cause frequent, high-magnitude confidence changes and answer flips, across
  prompting strategies, model sizes, and domains. Existing defences are largely ineffective or
  counterproductive. Subtle **semantics-preserving** edits produce misleading confidence.
- **Use:** Semantics-preserving is the operative phrase — this is the sharpest violation of the invariance
  condition.

---

## 4. C4 — Eliciting VC is itself harmful, and VC doesn't drive behaviour

### Vulnerability of LLMs' Stated Beliefs? LLMs Belief Resistance Check Through Strategic Persuasive Conversation Interventions
- **Authors:** (see arXiv listing)
- **arXiv:** [2601.13590](https://arxiv.org/abs/2601.13590) — Jan 2026
- **Role:** ⭐ The best C4 evidence, and counter-intuitive enough to be quotable. Six LLMs (GPT-4o-mini,
  Llama 3.3-70B, Llama 3.2-3B, Mistral 7B, Qwen 2.5-7B, Qwen 2.5-72B), three domains.
- **Key numbers:** VC prompting **significantly decreases** belief robustness in **12 of 18** model-dataset
  combinations. Qwen 2.5-72B is uniformly negative across **all 21** condition-dataset cells (−6.3 to
  −24.8 pp). Contrary to the human-psychology prediction.
- **Use:** The elicitation prompt is not a passive measurement. This is our strongest independent support
  for the "measurement perturbs the system" framing, and C4 is where we have the most room.

### Are LLM Decisions Faithful to Verbal Confidence?
- **Authors:** Jiawei Wang, Yanfei Zhou, et al. (USC)
- **arXiv:** [2601.07767](https://arxiv.org/abs/2601.07767) — 12 Jan 2026
- **Role:** ⚠️ Introduces **RiskEval** — does the model adjust its abstention policy as error penalties rise?
  Finding: models are neither cost-aware when verbalising confidence nor strategically responsive when
  deciding to abstain. Under extreme penalties where frequent abstention is mathematically optimal, models
  almost never abstain → utility collapse.
- **Caveat before citing:** the authors' own framing is that models *do* often "know" their uncertainty and
  that verbal confidence is useful/calibrated — the failure they diagnose is the confidence→policy link,
  not the confidence itself. Cite for the **invariance-to-risk / no-downstream-effect** condition; do not
  cite it as saying VC is uninformative.

---

## 5. Faithfulness — the signal exists internally, the verbalisation destroys it

### What LLM Forecasters Know but Don't Say: Probing Internal Representations for Calibration and Faithfulness
- **arXiv:** [2607.08046](https://arxiv.org/abs/2607.08046) — Jul 2026 ·
  [project page](https://www.pratyush.site/publication/llm_forecasters_probing/)
- **Role:** ⭐⭐ **The strongest single result for our position.** Lightweight probes on intermediate
  activations recover a well-calibrated confidence signal that the model's verbalisation *distorts*. The
  calibrated signal is present in the residual stream **from the final prompt token onward**, yet the
  stated probability does not report it.
- **Money line:** overconfidence is "less a failure of self-knowledge than of self-report."
- **Bonus finding:** forecasts are largely committed *before* reasoning begins — a single pre-reasoning pass
  recovers the answer and confidence; routing by pre-set answer spread saves 30–47% of tokens with no
  accuracy loss. (This also bounds the "CoT rescues VC" counterargument.)
- **Use:** Establishes that the verbalisation channel is lossy/distorting, not merely noisy. Pairs with
  2604.22215's R² < 0.01.

### LLM Doesn't Know What It Doesn't Know: Detecting Epistemic Blind Spots via Cross-Model Attribution Divergence on Clinical Tabular Data
- **arXiv:** [2606.19509](https://arxiv.org/abs/2606.19509) — 17 Jun 2026
- **Role:** ⚠️ **Read the caveat.** The headline VC numbers are real and strong: zero-shot VC = **0.856**
  regardless of whether SHAP evidence is injected (accuracy 49% vs 52%); few-shot = **0.937**; VC predicts
  errors at **AUROC = 0.50** (chance). Confidence tracks *prompt format*, not prediction quality —
  "not merely miscalibrated but completely invariant to prediction quality."
- **Caveat:** this is a clinical-tabular-data paper about epistemic blind spots and cross-model attribution
  divergence, not a general-purpose VC study. Cite the numbers with the domain stated; do not present it as
  a broad VC evaluation or reviewers will catch it.

### Can LLMs Introspect? A Reality Check
- **Authors:** Shashwat Singh, Tal Linzen, Shauli Ravfogel
- **arXiv:** [2605.26242](https://arxiv.org/abs/2605.26242) — 25 May 2026 (rev. 21 Aug 2026)
- **Role:** Two criteria for genuine introspection: **privileged access** and **second-order computation**.
  Re-examines two prior paradigms and finds input-only classifiers match the models' in-context
  predictions — earlier findings did not demonstrate access to internal representations. Models cannot
  distinguish tampered internal states from input manipulations (generic anomaly detection).
- **Verdict:** "current evidence is insufficient to establish metacognitive monitoring in LLMs."
- **Use:** Removes the implicit premise of the whole VC literature — that asking the model accesses
  something internal.

### Privileged Self-Access Matters for Introspection in AI
- **Authors:** Siyuan Song, Harvey Lederman, Jennifer Hu, Kyle Mahowald
- **arXiv:** [2508.14802](https://arxiv.org/abs/2508.14802) — 20 Aug 2025
- **Role:** Defines "thick" introspection (privileged self-access — more reliable access to its own
  workings than an external observer with comparable compute). LLMs reasoning about their own temperature
  appear to have lightweight introspection while failing the thick criterion.
- **Use:** Supplies the definition we need for the faithfulness necessary condition. The temperature
  experiment is the cleanest available analogy to VC.

### Anthropomimetic Uncertainty: What Verbalized Uncertainty in Language Models is Missing
- **Authors:** Dennis Ulmer, Alexandra Lorson, Ivan Titov, Christian Hardmeier
- **arXiv:** [2507.10587](https://arxiv.org/abs/2507.10587) — 11 Jul 2025 (rev. 20 Feb 2026)
- **Role:** Position paper on underexplored biases in verbalised uncertainty and the human-communication
  research VC ignores. Shares an author with 2601.08064.
- **Use:** Bridges to §6. Note it argues for *better* verbalisation rather than against the construct — a
  different destination from the same premises.

---

## 6. The formal contrast — what a real estimator looks like

### Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation
- **Authors:** Lorenz Kuhn, Yarin Gal, Sebastian Farquhar
- **arXiv:** [2302.09664](https://arxiv.org/abs/2302.09664) — 19 Feb 2023 · **ICLR 2023 (Spotlight)**
- **Role:** Entropy over a well-specified partition of output space (semantic equivalence classes).
  Invariant to paraphrase by construction. No retraining, no supervision.

### Detecting hallucinations in large language models using semantic entropy
- **Authors:** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal
- **Venue:** **Nature** 630(8017), 625–630, 2024 · DOI [10.1038/s41586-024-07421-0](https://doi.org/10.1038/s41586-024-07421-0)
  · [OATML post](https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting_hallucinations_2024.html)
- **Role:** The Nature version. Use for the "defined quantity, stated invariance" side of the contrast.

### A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification
- **Authors:** Anastasios N. Angelopoulos, Stephen Bates
- **arXiv:** [2107.07511](https://arxiv.org/abs/2107.07511) — 15 Jul 2021 (rev. Dec 2022)
- **Role:** ⭐ **The sharpest rhetorical contrast: conformal prediction has a theorem, VC has a prompt.**
  Distribution-free finite-sample guarantee P(y ∈ C(x)) ≥ 1 − α under exchangeability, for any pretrained
  model, without distributional assumptions.

### On Calibration of Modern Neural Networks
- **Authors:** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
- **arXiv:** [1706.04599](https://arxiv.org/abs/1706.04599) — 14 Jun 2017 · **ICML 2017**
- **Role:** ⚠️ **Discipline for C1.** Calibration *is* formally defined here (and reliability diagrams / ECE
  standardised). So "VC has no formal background" is too strong as stated.
- **The defensible C1:** VC has no **estimator-level** definition. There is no function of the model's
  parameters and input that VC approximates, so no consistency, convergence, or coverage result can attach
  to it. Calibration is a property VC can be *measured against*; it is not a definition of what VC *is*.

### Verification of Forecasts Expressed in Terms of Probability
- **Author:** Glenn W. Brier
- **Venue:** *Monthly Weather Review* 78(1), 1–3, 1950 ·
  DOI [10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)
- **Role:** Origin of the Brier score / proper scoring rules. Needed for the C1 discipline above and to read
  ConfTuner's tokenized-Brier claim.

### Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey
- **Authors:** Xiaoou Liu, Tiejin Chen, Longchao Da, Chacha Chen, Zhen Lin, Hua Wei
- **arXiv:** [2503.15850](https://arxiv.org/abs/2503.15850) — 20 Mar 2025 (rev. 3 Jun 2025)
- **Role:** Taxonomy reference for positioning VC against semantic entropy, conformal prediction, and
  logit-based methods. Categorises by computational efficiency and uncertainty dimension (input, reasoning,
  parameter, prediction).

### Human psychometrics of verbal probability (pre-2022, ignored by most of the VC literature)
- **Wallsten, Budescu, Rapoport, Zwick & Forsyth (1986),** "Measuring the vague meanings of probability
  terms," *J. Exp. Psychol.: General* 115(4), 348–365 ·
  DOI [10.1037/0096-3445.115.4.348](https://doi.org/10.1037/0096-3445.115.4.348)
- **Budescu & Wallsten (1990),** "Dyadic decisions with verbal and numerical probabilities,"
  *Organizational Behavior and Human Decision Processes* 46(2), 240–263.
- **Budescu & Wallsten (1995),** "Processing linguistic probabilities: General principles and empirical
  evidence," *Psychology of Learning and Motivation* 32, 275–318.
- **Windschitl & Wells (1996)** and the review literature on verbal vs. numeric probability, e.g.
  Wintle et al., "Verbal probabilities: Very likely to be somewhat more confusing than numbers," *PLOS ONE*
  2019 · [link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0213522)
- **Why it matters:** decades of evidence that verbal probability expressions are vague, inter-personally
  variable, and strongly context-dependent — and that people prefer to *receive* numbers but *give* words.
  Xiong et al. explicitly flag that LLMs may be "imitating human patterns of expressing confidence." If so,
  VC inherits a construct psychology already knows is unreliable. **This is unexploited territory** —
  almost none of the 2022+ VC papers cite it.

---

## 7. Counterarguments we must defend against

### CA1 — "VC is sometimes well-calibrated" → [2412.14737](https://arxiv.org/abs/2412.14737)
Reply: prompt-conditional calibration is not calibration. A property that holds only for a prompt found by
search over prompt space is overfitting to the eval set, not a property of an estimator. Reinforced by
2605.27752 (the comparison's sign is protocol-dependent) and 2601.08064 (calibration metrics are
independent of robustness/stability/sensitivity).

### CA2 — "It can be fixed by training"

#### ConfTuner: Training Large Language Models to Express Their Confidence Verbally
- **Authors:** Yibo Li, Miao Xiong, Jiaying Wu, Bryan Hooi
- **arXiv:** [2508.18847](https://arxiv.org/abs/2508.18847) — 26 Aug 2025 (rev. 25 Nov 2025) · **NeurIPS 2025**
- **Claim:** a **tokenized Brier score** loss, proved to be a proper scoring rule, correctly incentivises
  the model to report its true probability of correctness. No ground-truth confidence labels needed;
  transfers to GPT-4o, self-correction, model cascades.
- **Note:** the most serious CA2 paper — it supplies exactly the estimator-level grounding we say VC lacks.
  Our reply must be scoped: it grounds *ConfTuner-trained* VC, not VC-as-elicited, and the fix existing at
  all concedes that untrained VC fails. Check whether the properness argument survives the discretisation
  of the token vocabulary (this is a real technical opening — a proper scoring rule over a ~5-atom
  reachable support is proper over a degenerate simplex).

#### LoVeC: Reinforcement Learning for Better Verbalized Confidence in Long-Form Generation
- **arXiv:** [2505.23912](https://arxiv.org/abs/2505.23912) · **ACL 2026** —
  [ACL Anthology](https://aclanthology.org/2026.acl-long.1539/)
- RL-trained on-the-fly per-statement confidence for long-form generation; 20× faster than self-consistency
  with better calibration.

#### CritiCal: Can Critique Help LLM Uncertainty or Confidence Calibration?
- **arXiv:** [2510.24505](https://arxiv.org/abs/2510.24505)
- Natural-language critique-based calibration training; beats its GPT-4o teacher on complex reasoning.
  Distinguishes critiquing *uncertainty* (question-focused, better for open-ended) from *confidence*
  (answer-specific, better for MCQ).

> **Shared reply to CA2:** every one of these exists because untrained VC fails. They also all replace the
> elicited quantity with a *trained* one — which is conceding that the prompt-elicited number was never the
> estimator. Ask what the trained model's VC is an estimator *of*, and whether that survives the C3 tests.

### CA3 — "CoT rescues it"

#### Read Your Own Mind: Reasoning Helps Surface Self-Confidence Signals in LLMs
- **Authors:** Jakub Podolak, Rajeev Verma
- **arXiv:** [2505.23845](https://arxiv.org/abs/2505.23845) — 28 May 2025 (rev. 5 Nov 2025)
- **Claim:** in the default answer-then-confidence setting the model is regularly overconfident while
  semantic entropy stays reliable; under extended reasoning, verbal confidence improves substantially.
- **The concession inside it:** a *separate reader model that sees only the chain* reconstructs very similar
  confidences. So the improved score reflects the alternatives surfaced in the visible text, not privileged
  self-access — which is our faithfulness argument, and it dovetails with 2607.08046's finding that the
  answer is committed before reasoning begins.

---

## 8. Where the gap is (target contribution)

No one has written the unified negative result. Each paper frames its finding as a problem to *fix* rather
than as evidence the construct is unsound. The paper to write:

1. **Formalise** what VC would have to be to count as an estimator.
2. **Show it fails the necessary conditions:**
   - invariance to semantically-null prompt perturbation → 2601.08064, 2507.06489, 2503.02863
   - support density → 2603.09309, 2509.25532, 2604.22215
   - faithfulness to internal state → 2607.08046, 2604.22215, 2605.26242, 2508.14802
   - monotone response to risk → 2601.07767
   - *(candidate fifth)* protocol-identifiability → 2605.27752
3. **Demonstrate the ECE-under-discretisation pathology.** ⭐ Most publishable single result: when
   confidence has ~5 atoms of support, low ECE is achievable by a constant predictor and the metric is
   uninformative. 2603.09309 gestures at this ("potentially distorts ECE"); nobody has proved it.

**Open work not yet done:** pull the full citation graph around Lin / Tian / Xiong and quantify how many of
the 100+ downstream papers use VC as ground-truth uncertainty without validating it. Combine with §6's
psychometric literature for the "the field re-derived a known-bad human construct" framing.

---

## Appendix — corrections to the working notes

| Working note said | Actually |
|---|---|
| DINCO is a 2026 paper | arXiv Sep 2025 (2509.25532) |
| "LLM Doesn't Know What It Doesn't Know" is a general VC study | It is a clinical-tabular epistemic-blind-spot paper (2606.19509); the VC numbers are real but domain-specific |
| "Are LLM Decisions Faithful" says internal uncertainty is flat / invariant to risk | The paper's own claim is that verbal confidence is often calibrated but never converted into an abstention policy; the invariance is in the *decision*, not the confidence |
| "VC has no formal background" | Too strong — calibration and proper scoring rules are formally defined (Guo 2017, Brier 1950). Use "no estimator-level definition" |
| Rescaling Confidence: "multiples of five, three values" | Confirmed: >78% of responses on three round values, 6 LLMs × 3 datasets |
