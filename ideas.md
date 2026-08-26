

| Decision | Specification | Why |
| :--- | :--- | :--- |
| **Dataset** | `FERMI`, RealFP split — [github.com/allenai/fermi](https://github.com/allenai/fermi) | Fermi problems (*"how much would sea level rise if all ice melted"*) are questions where a **range** is the honest answer. Dispersion is meaningful, unlike arithmetic where there is one right number. |
| **Size** | 300 items for pilot, then all ~1k | RealFP is only ~1k total. |
| **Model** | `Llama-3.1-8B-Instruct`, 4-bit via HF | clone from HF|
| **Samples** | $k = 10$ per question, temperature 0.8 | Comparable results|
| **Quantile levels** | 0.1, 0.5, 0.9 | Three is enough to detect crossing and tail behaviour. More later if it works. |
| **Scale** | $\log_{10}$ throughout | Prevents large orders of magnitude from dominating averages. |
| **Contrast set** | GSM8K, 100 items (at the very end only) | Sharp-answer control. An interval around an arithmetic answer is semantically odd, which is the point of including it. |


## Some info

* **Work in Log Space:** Fermi answers span many orders of magnitude. Take $\log_{10}$ of every predicted and true value immediately at load time, and carry all quantiles, pinball loss, CRPS, and interval calculations in that space. Only convert back to raw units for visualization. Grade correctness as within a factor of 10:
  $$|\log_{10}\hat{y} - \log_{10}y| \le 1$$
  Filter out non-positive entries ($y \le 0$) during loading and report the exact drop count.
* **Quantiles as a Forecast:** Rather than eliciting single-point estimates, collect distributional claims ($\hat{q}_{0.1}, \hat{q}_{0.5}, \hat{q}_{0.9}$) to evaluate coverage and spread.
* **Pinball Loss:** Evaluates individual quantile accuracy and reveals asymmetric tail or median errors:
  $$L_\tau(\hat{q}, y) = (\tau - \mathbb{1}[y < \hat{q}])(y - \hat{q})$$
* **Continuous Ranked Probability Score (CRPS):** Evaluates overall predictive distributions without parametric assumptions, enabling a direct scoring comparison between verbalized quantiles and sampled ensembles:
  $$\mathrm{CRPS}(F,y) = \mathbb{E}|X-y| - \tfrac{1}{2} \mathbb{E}|X-X'|, \quad X, X' \sim F$$
* **Rank Histograms (Talagrand Diagrams):** Maps where ground-truth values fall within predicted ensembles. A flat profile indicates well-dispersed calibration, a U-shape indicates overconfidence (too narrow), and a domed shape indicates underconfidence (too wide).
* **Conformalized Quantile Regression (CQR):** Reference Romano, Patterson & Candès (2019) to compute valid distribution-free prediction intervals.



## Step-by-Step Task Breakdown

### 1. Dataset Ingestion and Characterisation
Pull the RealFP split from the AllenAI repository and map it into the shared JSON schema. Apply $\log_{10}$ transformations to all target values, remove non-positive records, and log the total number of filtered entries. Produce a concise one-page summary detailing surviving record counts, the empirical distribution of $\log_{10}$ ground truths, and raw model accuracy at a factor-of-10 margin on an initial 100-item batch.



### 2. Verbalized Quantile Elicitation
Query the model for the 10th, 50th, and 90th percentiles. Preserve state caching, retry mechanisms, and parsing checks. Quantile crossing events ($\hat{q}_{0.1} > \hat{q}_{0.5}$ or $\hat{q}_{0.5} > \hat{q}_{0.9}$) must be flagged and tallied as an empirical finding rather than artificially sorted. Track parsing failures in a separate log and evaluate prompt sensitivity across three distinct prompt phrasings.

### 3. Sampled-Ensemble Baseline Generation
Generate $k=10$ independent stochastic generations per prompt at temperature 0.8. Extract empirical quantiles in log space to measure the implicit uncertainty captured by the model's sampling distribution.

### 4. Pre-Registration and Pinball Loss Scoring
Before computing metric values, submit explicit pre-registered threshold criteria to your supervisor defining what constitutes a statistical match between verbalized and sampled distributions. Once approved, compute the pinball loss across both generation methods at $\tau \in \{0.1, 0.5, 0.9\}$ to isolate whether errors concentrate in the distribution tails or the median.

### 5. Sample-Based CRPS Computation
Calculate the CRPS for both the verbalized quantiles and sampled ensembles in log space. Sort sample arrays prior to evaluation to execute computations in $O(k \log k)$ time instead of naive pairwise loops.

### 6. Rank Histogram Evaluation
Construct Talagrand rank diagrams for both the ensemble and verbalized distributions. Evaluate whether the sampled ensemble displays a uniform rank profile while verbalized estimates show a U-shaped overconfidence pattern.

### 7. Conformal Calibration (CQR)
Apply conformal quantile regression to the verbalized intervals using the shared conformal utility in the main repository. Compare the resulting calibrated interval widths against conformalized ensemble intervals under identical coverage guarantees.

### 8. GSM8K Contrast and Final Synthesis
Run a 100-item control evaluation on GSM8K through the identical pipeline to evaluate interval behavior on deterministic arithmetic tasks. Compile findings into a self-contained three-page report covering methodology, empirical findings, and recommended next steps.
