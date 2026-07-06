"""
Approach A — Bootstrap confidence intervals on the external test set (N=421),
plus a PAIRED bootstrap comparison Odor-Sight vs Odorify.

Reviewer concern addressed: the external metrics were single point estimates.
Here we quantify their sampling uncertainty WITHOUT retraining the published
model: we regenerate its test-set probabilities (deterministic) and resample
the 421 test molecules with replacement (stratified) B times.

Honest threshold: the published headline used a threshold tuned on the test
set itself (G-mean on the test ROC). This script can instead derive the
operating threshold from the TRAINING data only (--threshold-method holdout|cv5)
and report all test metrics at that locked threshold. The test-tuned value and
the default 0.5 are also reported as sensitivity references.

Outputs (in ./results/, nothing is overwritten elsewhere):
  - test_predictions.csv         per-molecule y_true / Odor-Sight prob / Odorify prob,pred
  - bootstrap_summary.csv        point + 95% CI for every metric / model / threshold
  - paired_diff_summary.csv      Odor-Sight - Odorify difference + CI + bootstrap p
  - bootstrap_forest.png         visual summary
  - protocol_log.txt             parameters, seeds, reproduction check
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split

import common as C


def derive_honest_threshold(method, train_list, params, ndf, edim, device,
                            epochs, criterion, seed, log):
    """Return an operating threshold derived ONLY from training data."""
    if method == "fixed":
        return 0.5, "fixed-0.5"
    if method == "gmean_test":
        return None, "gmean_test"  # handled by caller using test probs (reference only)
    if method == "holdout":
        # single stratified 85/15 split of the training set
        labels = np.array([d.y.item() for d in train_list])
        tr_idx, va_idx = train_test_split(
            np.arange(len(train_list)), test_size=0.15,
            stratify=labels, random_state=seed)
        tr = [train_list[i] for i in tr_idx]
        va = [train_list[i] for i in va_idx]
        log(f"  [holdout] training aux model on {len(tr)} mols, picking threshold on {len(va)}")
        m = C.train_model(tr, params, ndf, edim, device, epochs=epochs, seed=seed)
        yv, pv = C.predict_probs(m, va, device)
        t = C.pick_threshold(yv, pv, criterion=criterion)
        return t, f"holdout-{criterion}"
    if method == "cv5":
        # 5-fold out-of-fold predictions on the training set
        labels = np.array([d.y.item() for d in train_list])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_true, oof_prob = [], []
        for k, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(train_list)), labels)):
            tr = [train_list[i] for i in tr_idx]
            va = [train_list[i] for i in va_idx]
            log(f"  [cv5] fold {k+1}/5: train {len(tr)} / val {len(va)}")
            m = C.train_model(tr, params, ndf, edim, device, epochs=epochs, seed=seed + k)
            yv, pv = C.predict_probs(m, va, device)
            oof_true.extend(yv); oof_prob.extend(pv)
        t = C.pick_threshold(np.array(oof_true), np.array(oof_prob), criterion=criterion)
        return t, f"cv5-{criterion}"
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--threshold-method", default="fixed",
                    choices=["fixed", "gmean_test", "holdout", "cv5"])
    ap.add_argument("--threshold-criterion", default="bacc",
                    choices=["bacc", "gmean", "youden"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = os.path.join(C.HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, "protocol_log.txt"), "w", encoding="utf-8")

    def log(msg=""):
        print(msg)
        logf.write(str(msg) + "\n")
        logf.flush()

    device = C.get_device()
    C.set_seed(args.seed)
    params = C.load_params()
    log(f"Device: {device} | params: {params}")
    log(f"Bootstrap B={args.n_boot} | threshold-method={args.threshold_method} "
        f"({args.threshold_criterion}) | seed={args.seed}")

    # --- Load data ---------------------------------------------------------
    df_train = pd.read_csv(os.path.join(C.DATA_DIR, "train_dataset.csv"))
    df_test = pd.read_csv(os.path.join(C.DATA_DIR, "test_dataset.csv"))

    # Replicate the original train/test overlap removal (canonical SMILES)
    df_train["canon"] = df_train["SMILES"].apply(C.canonicalize_smiles)
    df_test["canon"] = df_test["SMILES"].apply(C.canonicalize_smiles)
    df_train = df_train.dropna(subset=["canon"])
    df_test = df_test.dropna(subset=["canon"])
    overlap = set(df_train["canon"]) & set(df_test["canon"])
    if overlap:
        log(f"Removing {len(overlap)} train molecules overlapping with test (as in original).")
        df_train = df_train[~df_train["canon"].isin(overlap)]

    train_list, _, _ = C.df_to_data_list(df_train)
    test_list, test_smiles, y_true = C.df_to_data_list(df_test)
    ndf, edim = C.infer_dims(train_list)
    log(f"Train mols: {len(train_list)} | Test mols: {len(test_list)} "
        f"(of {len(df_test)} rows) | node_feats={ndf} edge_dim={edim}")
    log(f"Test class balance: pos={int(y_true.sum())} neg={int((y_true==0).sum())}")

    # --- Load published model and regenerate test probabilities ------------
    model = C.load_saved_model(params, ndf, edim, device,
                               os.path.join(C.MODELING, "gnn_best_model_cv5.pth"))
    y_true2, os_prob = C.predict_probs(model, test_list, device)
    assert np.array_equal(y_true, y_true2)

    # --- Reproduction check vs the published run ---------------------------
    b05 = C.metrics_at_threshold(y_true, os_prob, 0.5)["BACC"]
    t_gmean_test = C.pick_threshold(y_true, os_prob, "gmean")
    bgm = C.metrics_at_threshold(y_true, os_prob, t_gmean_test)["BACC"]
    log("\n--- Reproduction check (expected: BACC@0.5=0.8916, gmean thr~0.9376 -> BACC~0.9124) ---")
    log(f"  BACC@0.5          = {b05:.4f}")
    log(f"  gmean test thr    = {t_gmean_test:.4f}  -> BACC = {bgm:.4f}")

    # --- Load Odorify predictions and align by canonical SMILES ------------
    df_odo = pd.read_csv(os.path.join(C.MODELING, "..", "benchmark_odorify",
                                      "RUNS", "predicted_output.csv"))
    df_odo["canon"] = df_odo["SMILES"].apply(C.canonicalize_smiles)
    odo_map = {r["canon"]: (float(r["prob"]), int(round(float(r["pred_odor"]))))
               for _, r in df_odo.iterrows() if r["canon"] is not None}
    test_canon = [C.canonicalize_smiles(s) for s in test_smiles]
    have_odo = np.array([c in odo_map for c in test_canon])
    odo_prob = np.array([odo_map[c][0] if c in odo_map else np.nan for c in test_canon])
    odo_pred = np.array([odo_map[c][1] if c in odo_map else -1 for c in test_canon])
    log(f"\nOdorify predictions aligned for {int(have_odo.sum())}/{len(test_canon)} test molecules.")

    # --- Honest operating threshold ---------------------------------------
    t_honest, t_label = derive_honest_threshold(
        args.threshold_method, train_list, params, ndf, edim, device,
        args.epochs, args.threshold_criterion, args.seed, log)
    if t_honest is None:  # gmean_test reference
        t_honest, t_label = t_gmean_test, "gmean_test"
    log(f"\nPrimary (honest) threshold [{t_label}] = {t_honest:.4f}")

    # thresholds we will bootstrap for Odor-Sight
    os_thresholds = {
        f"primary[{t_label}]": t_honest,
        "ref-0.5": 0.5,
        "ref-gmean_test-0.9376": t_gmean_test,
    }

    # --- Save per-molecule predictions ------------------------------------
    pd.DataFrame({
        "SMILES": test_smiles, "canon": test_canon, "y_true": y_true,
        "odorsight_prob": os_prob, "odorify_prob": odo_prob, "odorify_pred": odo_pred,
    }).to_csv(os.path.join(outdir, "test_predictions.csv"), index=False)

    # --- Bootstrap ---------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    metrics = C.METRIC_ORDER
    boot = {f"OS|{lab}": {m: [] for m in metrics} for lab in os_thresholds}
    boot["Odorify"] = {m: [] for m in metrics}
    paired = {m: [] for m in metrics}  # OS primary - Odorify, on molecules with odorify preds

    idx_have = np.where(have_odo)[0]
    for _ in range(args.n_boot):
        bi = C.stratified_boot_indices(y_true, rng)
        yt = y_true[bi]
        # Odor-Sight at each threshold
        for lab, thr in os_thresholds.items():
            mm = C.metrics_at_threshold(yt, os_prob[bi], thr)
            for m in metrics:
                boot[f"OS|{lab}"][m].append(mm[m])
        # Odorify (paired on same resample, restricted to molecules we have)
        bi_h = bi[np.isin(bi, idx_have)]
        if len(np.unique(y_true[bi_h])) == 2:
            mo = C.metrics_from_pred(y_true[bi_h], odo_pred[bi_h], odo_prob[bi_h])
            mos = C.metrics_at_threshold(y_true[bi_h], os_prob[bi_h], t_honest)
            for m in metrics:
                boot["Odorify"][m].append(mo[m])
                paired[m].append(mos[m] - mo[m])

    # --- Point estimates (full sample) ------------------------------------
    point = {}
    for lab, thr in os_thresholds.items():
        point[f"OS|{lab}"] = C.metrics_at_threshold(y_true, os_prob, thr)
    point["Odorify"] = C.metrics_from_pred(y_true[idx_have], odo_pred[idx_have], odo_prob[idx_have])

    # --- Summaries ---------------------------------------------------------
    rows = []
    for key, md in boot.items():
        model_name, thr_label = (key.split("|", 1) + [""])[:2] if "|" in key else (key, "platform")
        for m in metrics:
            if len(md[m]) == 0:
                continue
            mean, median, lo, hi, std = C.percentile_ci(md[m])
            rows.append({
                "model": model_name, "threshold": thr_label, "metric": m,
                "point": point[key][m], "boot_mean": mean, "boot_median": median,
                "ci95_low": lo, "ci95_high": hi, "boot_std": std,
            })
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "bootstrap_summary.csv"), index=False)

    drows = []
    for m in metrics:
        if len(paired[m]) == 0:
            continue
        arr = np.asarray(paired[m], dtype=float)
        arr = arr[~np.isnan(arr)]
        mean, median, lo, hi, std = C.percentile_ci(arr)
        p_two = 2.0 * min((arr <= 0).mean(), (arr >= 0).mean())
        drows.append({
            "metric": m,
            "diff_point": point[f"OS|primary[{t_label}]"][m] - point["Odorify"][m],
            "diff_mean": mean, "ci95_low": lo, "ci95_high": hi,
            "prob_OS_superior": float((arr > 0).mean()),
            "p_bootstrap_two_sided": float(min(1.0, p_two)),
            "significant_95": bool(lo > 0 or hi < 0),
        })
    pd.DataFrame(drows).to_csv(os.path.join(outdir, "paired_diff_summary.csv"), index=False)

    # --- Console table -----------------------------------------------------
    log(f"\n================ Odor-Sight @ primary threshold [{t_label}={t_honest:.4f}] ================")
    log(f"{'metric':12} {'point':>8} {'95% CI':>22}")
    pk = f"OS|primary[{t_label}]"
    for m in metrics:
        s = [r for r in rows if r["model"] == "OS" and r["metric"] == m
             and r["threshold"] == f"primary[{t_label}]"]
        if s:
            r = s[0]
            log(f"{m:12} {r['point']:8.4f}   [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]")
    log("\n================ Paired difference (Odor-Sight - Odorify) ================")
    log(f"{'metric':12} {'diff':>8} {'95% CI':>22} {'p(2s)':>8}")
    for r in drows:
        log(f"{r['metric']:12} {r['diff_point']:8.4f}   "
            f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]  {r['p_bootstrap_two_sided']:.4f}")

    # --- Forest plot -------------------------------------------------------
    _forest_plot(rows, drows, t_label, outdir)
    log(f"\nSaved outputs to: {outdir}")
    logf.close()


def _forest_plot(rows, drows, t_label, outdir):
    show = ["BACC", "MCC", "F1", "Precision", "Recall", "ROC_AUC", "PR_AUC"]
    df = pd.DataFrame(rows)
    os_p = df[(df.model == "OS") & (df.threshold == f"primary[{t_label}]")].set_index("metric")
    odo = df[df.model == "Odorify"].set_index("metric")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(show))
    for off, sub, color, lab in [(-0.15, os_p, "tab:blue", "Odor-Sight"),
                                 (0.15, odo, "tab:orange", "Odorify")]:
        pts = [sub.loc[m, "point"] if m in sub.index else np.nan for m in show]
        lo = [sub.loc[m, "ci95_low"] if m in sub.index else np.nan for m in show]
        hi = [sub.loc[m, "ci95_high"] if m in sub.index else np.nan for m in show]
        err = np.array([np.array(pts) - np.array(lo), np.array(hi) - np.array(pts)])
        ax.errorbar(pts, y + off, xerr=err, fmt="o", color=color, label=lab, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(show); ax.invert_yaxis()
    ax.set_xlabel("Score (95% bootstrap CI)"); ax.set_xlim(0, 1.02)
    ax.grid(axis="x", alpha=0.3); ax.legend(loc="lower left")
    ax.set_title("External test set (N=421): metrics with 95% bootstrap CIs")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bootstrap_forest.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
