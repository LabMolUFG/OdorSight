# Exact manuscript edits (reviewer: CIs / repeated splits)

Five edits (A–E). For each: the **anchor** (text to find in the manuscript), the **action**,
and the **drop-in text**. All numbers use the honest, training-derived threshold (0.734),
so the headline shifts only marginally (BACC 0.91→0.90; MCC 0.79 unchanged) while removing
the test-set threshold leakage and adding CIs.

> Why the numbers move slightly: the published 0.91 used a threshold tuned on the test set
> itself. We re-selected it on training cross-validation only; BACC becomes 0.90 and every
> value now carries a CI. The Odor-Sight − Odorify gaps (BACC +0.18, MCC +0.22) are unchanged.

---

## EDIT A — METHODOLOGY ▸ "QSAR Modeling" (ADD two paragraphs)

**Anchor — find this sentence (end of the QSAR Modeling subsection):**
> "The best-performing configuration was retrained on the full training set and evaluated on
> the external test set using BACC, MCC, Precision, Recall, F1 score, and confusion matrices."

**Action: insert immediately AFTER it:**

> **Decision threshold.** The operating threshold for binary classification was selected using
> the training data only, as the probability cut-off maximizing balanced accuracy on the
> out-of-fold predictions of the 5-fold cross-validation, and was then held fixed for the
> external evaluation; the external test labels were never used for threshold selection.
>
> **Uncertainty quantification.** The uncertainty of the external-test estimates was
> characterized in two complementary ways. First, non-parametric bootstrap 95% confidence
> intervals (CIs) were computed for every reported metric by resampling the external test set
> (N = 421) with replacement, stratified by class, over 10,000 iterations and taking the 2.5th
> and 97.5th percentiles; the threshold-independent areas under the ROC and precision–recall
> curves (ROC-AUC and PR-AUC) were also computed to characterize discrimination irrespective
> of the decision threshold, and the significance of differences relative to the benchmark
> model was assessed by a paired bootstrap using the same resampled molecules for both models.
> Second, to confirm that performance did not depend on the particular partition, the entire
> stratified split-and-retrain procedure was repeated for 15 independent random seeds—each
> drawing the held-out test set from the benchmark-disjoint subset under class stratification
> and retraining the model with the fixed optimized hyperparameters—and the mean, standard
> deviation, and range of each metric across seeds were recorded.

---

## EDIT B — RESULTS AND DISCUSSION ▸ "Modeling" (REPLACE one sentence)

**Anchor — replace this sentence:**
> "The optimized model demonstrated high proficiency in distinguishing odorant from odorless
> compounds, achieving a BACC of 0.91, an F1 score of 0.93, and an MCC of 0.79, alongside a
> Precision of 0.96 and a Recall of 0.91."

**Replacement:**

> The optimized model demonstrated high proficiency in distinguishing odorant from odorless
> compounds. On the external test set it achieved a BACC of 0.90 (95% CI 0.87–0.94), an F1
> score of 0.94 (0.92–0.96), and an MCC of 0.79 (0.73–0.85), alongside a Precision of 0.95
> (0.93–0.97) and a Recall of 0.92 (0.89–0.95); discrimination was strong and independent of
> the decision threshold (ROC-AUC 0.95 [0.93–0.97], PR-AUC 0.98 [0.97–0.99]). Performance was
> stable across 15 repeated stratified splits (BACC 0.90 ± 0.02, MCC 0.78 ± 0.03,
> F1 0.93 ± 0.01), confirming that these estimates are not an artifact of a single partition.

*(Keep the following existing sentence — "These results indicate strong predictive
performance across both classes…" — unchanged.)*

---

## EDIT C — RESULTS AND DISCUSSION ▸ "Benchmarking and Comparative Analysis"

**Anchor — find:**
> "In contrast, Odor-Sight demonstrated superior overall robustness: BACC 0.91, MCC 0.79,
> Precision 0.96, Recall 0.91, and F1 Score 0.93. The substantial improvements in MCC (+0.22)
> and BACC (+0.18) are particularly significant."

**Action: (1) change the Odor-Sight values, (2) add a significance sentence.**

Change the values to:
> "…Odor-Sight demonstrated superior overall robustness: BACC 0.90, MCC 0.79, Precision 0.95,
> Recall 0.92, and F1 Score 0.94."

Then, immediately AFTER "…(+0.18) are particularly significant.", insert:

> These differences were statistically significant by a paired bootstrap on the identical
> molecules (BACC +0.18, 95% CI 0.13–0.22; MCC +0.22, 0.14–0.31; ROC-AUC +0.13, 0.09–0.17; all
> p < 0.001). The only metric favoring Odorify was recall (0.98 vs 0.92), achieved at the cost
> of a markedly lower specificity (0.48 vs 0.89), reflecting its tendency to over-predict the
> odorant class—precisely the behavior penalized by the balanced metrics on which Odor-Sight
> leads.

---

## EDIT D — ABSTRACT (REPLACE one clause)

**Anchor — replace:**
> "the underlying Graph Neural Network achieved a Balanced Accuracy (BACC) of 0.91 and a
> Matthews Correlation Coefficient (MCC) of 0.79 on an external validation set, demonstrating
> strong performance under class imbalance."

**Replacement:**

> the underlying Graph Neural Network achieved a Balanced Accuracy (BACC) of 0.90 (95% CI
> 0.87–0.94) and a Matthews Correlation Coefficient (MCC) of 0.79 (0.73–0.85) on an external
> validation set, with performance remaining stable across repeated data splits, demonstrating
> strong and reproducible performance under class imbalance.

---

## EDIT E — METHODOLOGY ▸ "QSAR Modeling" — fix the class-ratio claim (recommended)

**Anchor — replace:**
> "Stratification ensured that the class distribution (~3.5:1) was preserved across both
> subsets, preventing underrepresentation of the minority class in the test set."

**Why:** the realized ratios are ≈2.4:1 (test) and ≈5.9:1 (train), because the test set is
drawn from the benchmark-disjoint subset; "~3.5:1 preserved in both subsets" is inaccurate.

**Replacement:**

> Stratification by class preserved the odorant-majority balance within each subset,
> preventing underrepresentation of the minority (odorless) class in the test set.

---

## Optional drop-in tables

**Table 1.** External-test performance of Odor-Sight (N = 421), with 95% bootstrap CIs and the
distribution across 15 repeated stratified splits.

| Metric | Estimate (95% CI) | Repeated splits (mean ± SD) |
|---|---|---|
| BACC | 0.90 (0.87–0.94) | 0.90 ± 0.02 |
| MCC | 0.79 (0.73–0.85) | 0.78 ± 0.03 |
| Precision | 0.95 (0.93–0.97) | 0.96 ± 0.02 |
| Recall | 0.92 (0.89–0.95) | 0.90 ± 0.02 |
| Specificity | 0.89 (0.83–0.94) | 0.91 ± 0.04 |
| F1 | 0.94 (0.92–0.96) | 0.93 ± 0.01 |
| ROC-AUC | 0.95 (0.93–0.97) | 0.95 ± 0.01 |
| PR-AUC | 0.98 (0.97–0.99) | 0.98 ± 0.01 |

**Table 2.** Head-to-head on the identical external set (N = 421); 95% bootstrap CIs and
paired-bootstrap significance.

| Metric | Odor-Sight (95% CI) | Odorify (95% CI) | Δ (95% CI) | p |
|---|---|---|---|---|
| BACC | 0.90 (0.87–0.94) | 0.73 (0.68–0.77) | +0.18 (0.13–0.22) | <0.001 |
| MCC | 0.79 (0.73–0.85) | 0.57 (0.49–0.65) | +0.22 (0.14–0.31) | <0.001 |
| Precision | 0.95 (0.93–0.97) | 0.82 (0.80–0.85) | +0.13 (0.10–0.16) | <0.001 |
| Recall | 0.92 (0.89–0.95) | 0.98 (0.96–0.99) | −0.05 (−0.08, −0.03) | <0.001 |
| Specificity | 0.89 (0.83–0.94) | 0.48 (0.39–0.57) | +0.41 (0.32–0.50) | <0.001 |
| F1 | 0.94 (0.92–0.96) | 0.89 (0.87–0.91) | +0.05 (0.02–0.07) | <0.001 |
| ROC-AUC | 0.95 (0.93–0.97) | 0.82 (0.78–0.87) | +0.13 (0.09–0.17) | <0.001 |
| PR-AUC | 0.98 (0.97–0.99) | 0.89 (0.86–0.92) | +0.09 (0.07–0.12) | <0.001 |

---

## Numbers to update for consistency (search-and-replace checklist)

The Odor-Sight external metrics appear in 3 places — keep them identical everywhere:
- Abstract (EDIT D): BACC 0.91 → **0.90** (+CI), MCC keeps 0.79 (+CI).
- Results/Modeling (EDIT B): 0.91/0.93/0.79/0.96/0.91 → **0.90/0.94/0.79/0.95/0.92** (+CIs).
- Benchmarking (EDIT C): 0.91/0.79/0.96/0.91/0.93 → **0.90/0.79/0.95/0.92/0.94**.
- The deltas in the Benchmarking sentence (MCC +0.22, BACC +0.18) **do not change**.
