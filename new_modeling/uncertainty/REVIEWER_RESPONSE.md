# Reviewer response — uncertainty of the external-test metrics

**Reviewer comment.** *"The external test set performance (BACC = 0.91, MCC = 0.79) is
reported from a single random stratified split (N = 421). No confidence intervals, no
repeated splits, no bootstrapping. Authors must report variance across multiple splits or
at minimum bootstrap confidence intervals on all reported metrics."*

We agree. We added two complementary analyses and (separately) corrected a minor
threshold-selection issue uncovered while doing so. All code is in
`OdorSight/new_modeling/uncertainty/`. Nothing in the original pipeline, model, threshold, or
datasets was overwritten.

---

## 0. Faithful reproduction first

Loading the published model (`gnn_best_model_cv5.pth`) and re-scoring the external test
set reproduces the original run **exactly**:

| Quantity | Original run | This reproduction |
|---|---|---|
| BACC @ threshold 0.5 | 0.8916 | **0.8916** |
| G-mean threshold (from test ROC) | 0.9376 | **0.9376** |
| BACC @ 0.9376 | 0.9124 | **0.9124** |

So the uncertainty analysis below is built on the same model that produced the published
numbers.

---

## 1. Bootstrap confidence intervals (Approach A)

Holding the trained model fixed, we resampled the external test set (N = 421) **with
replacement, stratified by class** (298 odorant / 123 odorless preserved), B = 10,000
iterations, and recomputed every metric per resample (95% percentile CIs). We also report
two **threshold-independent** metrics (ROC-AUC, PR-AUC).

### 1a. Threshold-selection fix (important)
The published headline metrics were computed at a decision threshold (0.9376) chosen by
**maximizing G-mean on the external test set itself** — i.e., the threshold "saw" the test
labels, a mild optimistic bias. We re-derived the threshold using **training data only**
(5-fold out-of-fold CV, maximizing balanced accuracy → t* = 0.734) and report the test
metrics at that locked threshold. **The headline barely changes**, which is reassuring:

| Metric | Published (test-tuned thr 0.9376) | **Honest thr 0.734** | 95% bootstrap CI (honest) |
|---|---|---|---|
| BACC | 0.91 | **0.904** | [0.872, 0.935] |
| MCC | 0.79 | **0.793** | [0.729, 0.854] |
| Precision | 0.96 | **0.952** | [0.928, 0.974] |
| Recall | 0.91 | **0.923** | [0.893, 0.953] |
| Specificity | — | **0.886** | [0.829, 0.943] |
| F1 | 0.93 | **0.937** | [0.916, 0.956] |
| ROC-AUC | — | **0.952** | [0.930, 0.972] |
| PR-AUC | — | **0.980** | [0.970, 0.989] |

For completeness, the **exact published numbers** (test-tuned threshold 0.9376) also now
carry CIs: BACC 0.912 [0.882, 0.940], MCC 0.795 [0.735, 0.853], Precision 0.964
[0.943, 0.985], Recall 0.906 [0.872, 0.936], F1 0.934 [0.913, 0.954]. And at the default
threshold 0.5: BACC 0.892 [0.855, 0.925], MCC 0.778 [0.710, 0.842], F1 0.934 [0.914, 0.953].

*(Full table: `results/bootstrap_summary.csv`; figure: `results/bootstrap_forest.png`.)*

### 1b. Benchmark comparison with CIs + significance (paired bootstrap)
Because Odorify's per-molecule predictions on the identical 421 compounds are available,
we ran a **paired** bootstrap (same resampled molecules for both tools). Odor-Sight's
advantage is statistically significant on every metric except recall (where Odorify scores
higher only because it labels almost everything "odorant" — its specificity is 0.48):

| Metric | Odor-Sight (honest) | Odorify | Δ (OS − Odorify) | 95% CI of Δ | p (two-sided) |
|---|---|---|---|---|---|
| BACC | 0.904 | 0.728 [0.683, 0.773] | **+0.176** | [0.131, 0.223] | < 0.001 |
| MCC | 0.793 | 0.571 [0.485, 0.651] | **+0.222** | [0.140, 0.311] | < 0.001 |
| Precision | 0.952 | 0.820 [0.795, 0.846] | **+0.132** | [0.104, 0.160] | < 0.001 |
| Recall | 0.923 | 0.977 [0.960, 0.993] | −0.054 | [−0.084, −0.027] | < 0.001 |
| Specificity | 0.886 | 0.480 [0.390, 0.569] | **+0.407** | [0.317, 0.496] | < 0.001 |
| F1 | 0.937 | 0.891 [0.874, 0.909] | **+0.046** | [0.024, 0.067] | < 0.001 |
| ROC-AUC | 0.952 | 0.824 [0.780, 0.866] | **+0.128** | [0.090, 0.169] | < 0.001 |
| PR-AUC | 0.980 | 0.889 [0.861, 0.916] | **+0.091** | [0.066, 0.117] | < 0.001 |

*(Full table: `results/paired_diff_summary.csv`.)*

---

## 2. Variance across repeated splits (Approach B)

We repeated the *original constrained* split (test set drawn only from the
benchmark-disjoint subset, N = 1,020, stratified by Outcome; training set = the remaining
exclusive molecules + all 3,192 benchmark-overlapping molecules) under **K = 15 independent
seeds**. For each seed the model was retrained from scratch with the fixed published
hyperparameters (80 epochs), the threshold was selected on a training-internal validation
partition (never the test set), and metrics were computed on that seed's held-out test set
(298 odorant / 123 odorless, as in the original).

**Performance is highly stable across splits, and the published single-split values fall in
the middle of these distributions:**

| Metric | Mean ± SD (K=15) | 95% interval | min – max |
|---|---|---|---|
| BACC | **0.904 ± 0.018** | [0.876, 0.930] | 0.875 – 0.931 |
| MCC | **0.780 ± 0.032** | [0.731, 0.829] | 0.726 – 0.833 |
| Precision | 0.959 ± 0.016 | [0.929, 0.980] | 0.926 – 0.981 |
| Recall | 0.903 ± 0.023 | [0.870, 0.932] | 0.866 – 0.933 |
| Specificity | 0.905 ± 0.041 | [0.827, 0.957] | 0.821 – 0.959 |
| F1 | **0.930 ± 0.011** | [0.912, 0.946] | 0.911 – 0.946 |
| ROC-AUC | **0.951 ± 0.011** | [0.932, 0.970] | 0.929 – 0.975 |
| PR-AUC | **0.976 ± 0.007** | [0.965, 0.988] | 0.964 – 0.990 |

At the fixed default threshold 0.5 the picture is identical and even slightly tighter
(BACC 0.903 ± 0.016, MCC 0.787 ± 0.030, F1 0.934 ± 0.011): the model's predictions are
sharply bimodal, so performance is insensitive to the exact threshold (this also explains
why the per-split optimal threshold varies — it is weakly identified, while the metrics are
not).

Runtime ≈ 4.7 min/seed on an RTX 2060 (~71 min for K=15). Re-running `repeated_splits.py
--seeds 30` resumes and adds more seeds (per-seed results are written incrementally).

Table: `results/repeated_splits_summary.csv` · Figure: `results/repeated_splits_distribution.png`

---

## 3. Proposed manuscript text

### Methods — add to "QSAR Modeling"
> *Decision threshold.* The operating threshold was selected using the training data only,
> as the value maximizing balanced accuracy on 5-fold cross-validation out-of-fold
> predictions, and then held fixed for the external evaluation; the external test labels
> were never used for threshold selection.
>
> *Uncertainty quantification.* Two complementary analyses characterized the uncertainty of
> the external-test estimates. First, non-parametric bootstrap 95% confidence intervals were
> computed for every metric by resampling the external test set (N = 421) with replacement,
> stratified by class, over 10,000 iterations (2.5th–97.5th percentiles); threshold-independent
> metrics (ROC-AUC, PR-AUC) were included to characterize discrimination irrespective of the
> decision threshold, and a paired bootstrap on the identical molecules was used to assess the
> significance of differences versus the benchmark model. Second, the entire stratified
> splitting and retraining procedure was repeated under K = 15 independent random seeds, and
> the mean, standard deviation, and 95% interval of each metric across splits are reported.

### Results — replace the single sentence of point estimates
> The optimized model distinguished odorant from odorless compounds with a balanced accuracy
> of 0.90 (95% CI 0.87–0.94), an F1 score of 0.94 (0.92–0.96), and an MCC of 0.79 (0.73–0.85),
> alongside a precision of 0.95 (0.93–0.97) and a recall of 0.92 (0.89–0.95); discrimination
> was strong and threshold-independent (ROC-AUC 0.95 [0.93–0.97], PR-AUC 0.98 [0.97–0.99]).
> Performance was stable across K = 15 repeated stratified splits (BACC = 0.90 ± 0.02,
> MCC = 0.78 ± 0.03, F1 = 0.93 ± 0.01), confirming the result is not specific to a single
> partition.

### Results — benchmarking sentence
> On the identical held-out set, Odor-Sight significantly outperformed Odorify (paired
> bootstrap): BACC +0.18 (95% CI 0.13–0.22), MCC +0.22 (0.14–0.31), and ROC-AUC +0.13
> (0.09–0.17), all p < 0.001.

### Minor wording
The text states stratification preserved "~3.5:1" in both subsets; the realized ratios are
≈2.4:1 (test) and ≈5.9:1 (train), because the test set is drawn from the benchmark-disjoint
subset. Suggest rephrasing to "stratification preserved the odorant-majority class balance
within each subset."

---

## 4. Files produced (none overwrite originals)
```
uncertainty/
  common.py                      # model (verbatim) + metrics + bootstrap + training helpers
  uncertainty_bootstrap.py       # Approach A
  repeated_splits.py             # Approach B
  results/
    test_predictions.csv         # per-molecule y_true, Odor-Sight prob, Odorify prob/pred
    bootstrap_summary.csv        # point + 95% CI, every metric/threshold/model
    paired_diff_summary.csv      # OS − Odorify, CI + bootstrap p
    bootstrap_forest.png
    repeated_splits_per_seed.csv # (Approach B, written incrementally)
    repeated_splits_summary.csv
    repeated_splits_distribution.png
    protocol_log.txt
```
