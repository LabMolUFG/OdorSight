"""
Runner for the classical ML baselines (RF/SVM/XGBoost) x (ECFP4 / core descriptors),
mirroring the GNN protocol on the new curation (4201). Produces, per (algo, feature):
  - tuned hyperparameters (5-fold CV BACC);
  - primary external-test metrics with bootstrap 95% CIs (honest threshold);
  - variance across K=15 benchmark-restricted repeated splits (mean +/- SD);
and a PAIRED bootstrap comparison of each classical model vs the GNN on the identical
primary test set (threshold-free ROC-AUC/PR-AUC + BACC/MCC/F1 at a common 0.5 cutoff).

Outputs -> ./results/
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import ml_common as M
import common as C   # made importable via ml_common's sys.path insert

OUT = os.path.join(M.HERE, "results"); os.makedirs(OUT, exist_ok=True)
ALGOS = ["rf", "svm", "xgb"]
FEATURES = ["ecfp4", "desc"]
SEEDS_K = list(range(1000, 1015))      # SAME seeds as the GNN K=15
GNN_THR = 0.9435                        # GNN honest threshold on new curation (Approach A)

logf = open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8")
def log(m=""):
    print(m); logf.write(str(m) + "\n"); logf.flush()


def compute_gnn_probs(pool, te_idx):
    """GNN new-curation model probabilities on the primary test molecules, keyed by InChIKey."""
    sub = pool.loc[te_idx]
    data, iks = [], []
    for ik, smi, lab in zip(sub["InChIKey"], sub["SMILES"], sub["Outcome"]):
        d = C.smiles_to_data(smi, label=int(lab))
        if d is not None:
            data.append(d); iks.append(ik)
    params = C.load_params(); ndf, edim = C.infer_dims(data); dev = C.get_device()
    gnn = C.load_saved_model(params, ndf, edim, dev, M.GNN_MODEL)
    _, p = C.predict_probs(gnn, data, dev)
    return dict(zip(iks, p))


def bootstrap_ci(y_true, prob, thr, n_boot, seed):
    rng = np.random.default_rng(seed)
    acc = {m: [] for m in C.METRIC_ORDER}
    for _ in range(n_boot):
        bi = C.stratified_boot_indices(y_true, rng)
        mm = C.metrics_at_threshold(y_true[bi], prob[bi], thr)
        for m in C.METRIC_ORDER: acc[m].append(mm[m])
    return {m: C.percentile_ci(acc[m]) for m in C.METRIC_ORDER}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--trials-rf", type=int, default=40)
    ap.add_argument("--trials-xgb", type=int, default=40)
    ap.add_argument("--trials-svm", type=int, default=20)
    args = ap.parse_args()
    n_trials = {"rf": args.trials_rf, "xgb": args.trials_xgb, "svm": args.trials_svm}

    log("Building pool + features (new curation 4201)...")
    pool = M.build_pool()
    y = pool["Outcome"].to_numpy().astype(int)
    X = {f: M.featurize(pool, f)[0] for f in FEATURES}
    log(f"Pool {len(pool)} | ECFP4 {X['ecfp4'].shape} | desc {X['desc'].shape} "
        f"| class balance {dict(zip(*np.unique(y, return_counts=True)))}")

    tr_idx, te_idx = M.primary_split_indices(pool)
    log(f"Primary split (seed 42): train {len(tr_idx)} / test {len(te_idx)}")

    log("Computing GNN probabilities on the primary test set...")
    gnn_by_ik = compute_gnn_probs(pool, te_idx)
    te_ik = pool.loc[te_idx, "InChIKey"].to_numpy()
    gnn_prob = np.array([gnn_by_ik.get(k, np.nan) for k in te_ik])
    yte = y[te_idx]

    point_rows, ci_rows, rep_rows, paired_rows, best_params = [], [], [], [], {}

    for algo in ALGOS:
        for feat in FEATURES:
            tag = f"{algo}|{feat}"
            log(f"\n===== {tag} =====")
            Xf = X[feat]
            Xtr, ytr = Xf[tr_idx], y[tr_idx]
            Xte = Xf[te_idx]

            best, cvbacc = M.tune(algo, Xtr, ytr, n_trials[algo])
            best_params[tag] = {"params": best, "cv5_bacc": cvbacc}
            log(f"  tuned (CV5 BACC={cvbacc:.4f}): {best}")

            thr = M.oof_threshold(algo, best, Xtr, ytr)
            model = M.make_model(algo, best, ytr); model.fit(Xtr, ytr)
            prob = model.predict_proba(Xte)[:, 1]
            pt = C.metrics_at_threshold(yte, prob, thr)
            log(f"  honest thr={thr:.3f} -> BACC={pt['BACC']:.4f} MCC={pt['MCC']:.4f} "
                f"F1={pt['F1']:.4f} ROC-AUC={pt['ROC_AUC']:.4f}")

            cis = bootstrap_ci(yte, prob, thr, args.n_boot, seed=42)
            for m in C.METRIC_ORDER:
                mean, med, lo, hi, sd = cis[m]
                point_rows.append({"algo": algo, "feature": feat, "metric": m, "point": pt[m]})
                ci_rows.append({"algo": algo, "feature": feat, "metric": m,
                                "point": pt[m], "ci95_low": lo, "ci95_high": hi})

            # --- paired vs GNN on the identical primary test (0.5 cutoff + AUCs) ---
            mask = ~np.isnan(gnn_prob)
            rng = np.random.default_rng(7)
            diffs = {m: [] for m in C.METRIC_ORDER}
            yg, pc, pg = yte[mask], prob[mask], gnn_prob[mask]
            for _ in range(args.n_boot):
                bi = C.stratified_boot_indices(yg, rng)
                mc = C.metrics_at_threshold(yg[bi], pc[bi], 0.5)
                mg = C.metrics_at_threshold(yg[bi], pg[bi], 0.5)
                for m in C.METRIC_ORDER: diffs[m].append(mc[m] - mg[m])
            for m in C.METRIC_ORDER:
                arr = np.array(diffs[m]); arr = arr[~np.isnan(arr)]
                mean, med, lo, hi, sd = C.percentile_ci(arr)
                p2 = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
                paired_rows.append({"algo": algo, "feature": feat, "metric": m,
                                    "delta_classical_minus_gnn": mean, "ci95_low": lo,
                                    "ci95_high": hi, "p_two_sided": min(1.0, p2)})

            # --- K=15 repeated constrained splits (fixed tuned HPs) ---
            per_seed = []
            for seed in SEEDS_K:
                tri, tei = M.constrained_split(pool, seed)
                ti, vi = train_test_split(tri, test_size=0.15, stratify=y[tri], random_state=seed)
                m2 = M.make_model(algo, best, y[ti]); m2.fit(Xf[ti], y[ti])
                pv = m2.predict_proba(Xf[vi])[:, 1]; t = C.pick_threshold(y[vi], pv, "bacc")
                ptv = m2.predict_proba(Xf[tei])[:, 1]
                per_seed.append(C.metrics_at_threshold(y[tei], ptv, t))
            for m in C.METRIC_ORDER:
                vals = np.array([d[m] for d in per_seed])
                mean, med, lo, hi, sd = C.percentile_ci(vals)
                rep_rows.append({"algo": algo, "feature": feat, "metric": m,
                                 "mean": mean, "std": sd, "ci95_low": lo, "ci95_high": hi})
            log(f"  K=15 BACC {np.mean([d['BACC'] for d in per_seed]):.4f} "
                f"+/- {np.std([d['BACC'] for d in per_seed], ddof=1):.4f}")

            # incremental saves
            pd.DataFrame(ci_rows).to_csv(os.path.join(OUT, "primary_bootstrap_CIs.csv"), index=False)
            pd.DataFrame(rep_rows).to_csv(os.path.join(OUT, "repeated_splits.csv"), index=False)
            pd.DataFrame(paired_rows).to_csv(os.path.join(OUT, "paired_vs_gnn.csv"), index=False)
            json.dump(best_params, open(os.path.join(OUT, "best_params.json"), "w"), indent=2)

    build_comparison(rep_rows, paired_rows, log)
    logf.close()


def build_comparison(rep_rows, paired_rows, log):
    """GNN-vs-classical master table (K=15 mean±SD) + comparison figure."""
    gnn = {"BACC": (0.886, 0.011), "MCC": (0.745, 0.022), "F1": (0.926, 0.009),
           "ROC_AUC": (0.949, 0.008), "PR_AUC": (0.978, 0.006)}
    rep = pd.DataFrame(rep_rows)
    show = ["BACC", "MCC", "F1", "ROC_AUC", "PR_AUC"]
    rows = [{"model": "GNN (GAT)", **{m: f"{gnn[m][0]:.3f} ± {gnn[m][1]:.3f}" for m in show}}]
    for algo in ALGOS:
        for feat in FEATURES:
            r = {"model": f"{algo.upper()} ({feat})"}
            for m in show:
                sub = rep[(rep.algo == algo) & (rep.feature == feat) & (rep.metric == m)]
                r[m] = f"{sub['mean'].iloc[0]:.3f} ± {sub['std'].iloc[0]:.3f}" if len(sub) else "-"
            rows.append(r)
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(OUT, "comparison_GNN_vs_classical.csv"), index=False)
    log("\n===== COMPARISON (K=15 mean ± SD) =====")
    log(comp.to_string(index=False))

    # figure: BACC mean±SD per model
    fig, ax = plt.subplots(figsize=(9, 5))
    labels, means, sds = [], [], []
    labels.append("GNN (GAT)"); means.append(gnn["BACC"][0]); sds.append(gnn["BACC"][1])
    for algo in ALGOS:
        for feat in FEATURES:
            sub = rep[(rep.algo == algo) & (rep.feature == feat) & (rep.metric == "BACC")]
            if len(sub):
                labels.append(f"{algo.upper()}\n{feat}"); means.append(sub['mean'].iloc[0]); sds.append(sub['std'].iloc[0])
    colors = ["tab:green"] + ["tab:blue", "tab:cyan"] * 3
    ax.bar(range(len(labels)), means, yerr=sds, capsize=4, color=colors[:len(labels)])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("BACC (mean ± SD, K=15 splits)"); ax.set_ylim(0.5, 1.0)
    ax.axhline(gnn["BACC"][0], ls="--", color="tab:green", alpha=0.5)
    ax.set_title("New curation: GNN vs classical ML baselines (BACC)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "comparison_BACC.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
