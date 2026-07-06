"""
Approach B — Variance of external-test performance across REPEATED stratified
splits (reviewer concern: results came from a single random split, seed=42).

For each of K seeds we reproduce the ORIGINAL constrained split logic:
  * the test set (~421) is drawn ONLY from the benchmark-exclusive molecules
    (no_overlap == True, N=1020), stratified by Outcome;
  * the training set = the remaining exclusive molecules + ALL benchmark-
    overlapping molecules (N=3192).
We then retrain the model with the FIXED published hyperparameters, choose the
operating threshold on a TRAINING-internal validation split (never the test
set), and evaluate on that seed's test set. Reporting mean +/- std and 95% CI
across seeds shows the published result is not an artefact of seed 42.

Speed: every molecule's graph is featurized ONCE and reused across splits.
Results are written incrementally to results/repeated_splits_per_seed.csv so a
long run can be interrupted/resumed safely.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import common as C

N_TEST = 421  # match the original external-test size


def featurize_pool(log):
    """Featurize the full curated pool (train+test) exactly once."""
    df_tr = pd.read_csv(os.path.join(C.DATA_DIR, "train_dataset.csv"))
    df_te = pd.read_csv(os.path.join(C.DATA_DIR, "test_dataset.csv"))
    pool = pd.concat([df_tr, df_te], ignore_index=True)
    log(f"Pool: {len(pool)} molecules | no_overlap=True: {int(pool['no_overlap'].sum())}")

    data, canon = [], []
    for s, l in zip(pool["SMILES"], pool["Outcome"]):
        d = C.smiles_to_data(s, label=int(l))
        data.append(d)
        canon.append(C.canonicalize_smiles(s))
    valid = np.array([d is not None for d in data])
    pool = pool.reset_index(drop=True)
    pool["valid"] = valid
    pool["canon"] = canon
    pool["_idx"] = np.arange(len(pool))
    log(f"Featurized OK: {int(valid.sum())}/{len(pool)}")
    return pool, data


def run_one_seed(seed, pool, data, params, ndf, edim, device, epochs, criterion, log):
    excl = pool[(pool["no_overlap"]) & (pool["valid"])]
    over = pool[(~pool["no_overlap"]) & (pool["valid"])]

    test_prop = N_TEST / len(excl)
    tr_excl, te = train_test_split(
        excl, test_size=test_prop, stratify=excl["Outcome"], random_state=seed)

    train_df = pd.concat([tr_excl, over], ignore_index=True)
    test_df = te.copy()

    # Replicate canonical-SMILES overlap removal (train vs test)
    test_canon = set(test_df["canon"])
    train_df = train_df[~train_df["canon"].isin(test_canon)]

    train_idx = train_df["_idx"].to_numpy()
    test_idx = test_df["_idx"].to_numpy()
    train_labels = pool.loc[train_idx, "Outcome"].to_numpy()

    # Honest threshold: one stratified 85/15 split of THIS seed's training set;
    # train on 85%, choose threshold on the held-out 15%, evaluate on test.
    tr_i, va_i = train_test_split(
        train_idx, test_size=0.15, stratify=train_labels, random_state=seed)
    train_list = [data[i] for i in tr_i]
    val_list = [data[i] for i in va_i]
    test_list = [data[i] for i in test_idx]

    model = C.train_model(train_list, params, ndf, edim, device, epochs=epochs, seed=seed)
    yv, pv = C.predict_probs(model, val_list, device)
    thr = C.pick_threshold(yv, pv, criterion=criterion)

    yt, pt = C.predict_probs(model, test_list, device)
    m_honest = C.metrics_at_threshold(yt, pt, thr)
    m_05 = C.metrics_at_threshold(yt, pt, 0.5)

    row = {"seed": seed, "threshold": thr,
           "n_train": len(tr_i), "n_val": len(va_i), "n_test": len(test_idx),
           "test_pos": int(yt.sum()), "test_neg": int((yt == 0).sum())}
    for m in C.METRIC_ORDER:
        row[m] = m_honest[m]                 # at honest threshold
        if m in ("BACC", "MCC", "F1"):
            row[f"{m}@0.5"] = m_05[m]         # reference at 0.5
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30, help="number of repeated splits")
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--threshold-criterion", default="bacc",
                    choices=["bacc", "gmean", "youden"])
    args = ap.parse_args()

    outdir = os.path.join(C.HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    logpath = os.path.join(outdir, "repeated_splits_log.txt")
    per_seed_path = os.path.join(outdir, "repeated_splits_per_seed.csv")
    logf = open(logpath, "a", encoding="utf-8")

    def log(msg=""):
        print(msg)
        logf.write(str(msg) + "\n")
        logf.flush()

    device = C.get_device()
    params = C.load_params()
    log(f"\n==== Repeated splits: {args.seeds} seeds from {args.start_seed} | "
        f"epochs={args.epochs} | threshold={args.threshold_criterion} | device={device} ====")

    pool, data = featurize_pool(log)
    train_list_dims = [d for d in data if d is not None]
    ndf, edim = C.infer_dims(train_list_dims)
    log(f"node_feats={ndf} edge_dim={edim}")

    # Resume support: skip seeds already present
    done = set()
    if os.path.exists(per_seed_path):
        try:
            done = set(pd.read_csv(per_seed_path)["seed"].tolist())
            log(f"Found {len(done)} completed seeds; will skip them.")
        except Exception:
            pass

    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    t0 = time.time()
    n_done = 0
    for k, seed in enumerate(seeds):
        if seed in done:
            continue
        ts = time.time()
        row = run_one_seed(seed, pool, data, params, ndf, edim, device,
                           args.epochs, args.threshold_criterion, log)
        dt = time.time() - ts
        n_done += 1
        # append incrementally
        header = not os.path.exists(per_seed_path)
        pd.DataFrame([row]).to_csv(per_seed_path, mode="a", header=header, index=False)
        eta = (time.time() - t0) / n_done * (len([s for s in seeds if s not in done]) - n_done)
        log(f"seed {seed} ({k+1}/{len(seeds)}): BACC={row['BACC']:.4f} "
            f"MCC={row['MCC']:.4f} F1={row['F1']:.4f} thr={row['threshold']:.3f} "
            f"[{dt:.0f}s] ETA ~{eta/60:.1f} min")

    summarize(per_seed_path, outdir, log)
    logf.close()


def summarize(per_seed_path, outdir, log):
    df = pd.read_csv(per_seed_path)
    log(f"\n==== Summary across {len(df)} seeds ====")
    rows = []
    cols = [c for c in df.columns if c in C.METRIC_ORDER or c.endswith("@0.5") or c == "threshold"]
    log(f"{'metric':14} {'mean':>8} {'std':>8} {'95% CI':>22} {'min':>8} {'max':>8}")
    for c in cols:
        vals = df[c].dropna().to_numpy()
        if len(vals) == 0:
            continue
        mean, median, lo, hi, std = C.percentile_ci(vals)
        rows.append({"metric": c, "n": len(vals), "mean": mean, "std": std,
                     "median": median, "ci95_low": lo, "ci95_high": hi,
                     "min": float(vals.min()), "max": float(vals.max())})
        log(f"{c:14} {mean:8.4f} {std:8.4f}   [{lo:.4f}, {hi:.4f}]  {vals.min():8.4f} {vals.max():8.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "repeated_splits_summary.csv"), index=False)

    # Distribution plot for the headline metrics
    show = [m for m in ["BACC", "MCC", "F1", "Precision", "Recall", "ROC_AUC", "PR_AUC"]
            if m in df.columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([df[m].dropna() for m in show], labels=show, showmeans=True)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"External-test metrics across {len(df)} repeated splits")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "repeated_splits_distribution.png"), dpi=150)
    plt.close(fig)
    log(f"Saved summary + plot to {outdir}")


if __name__ == "__main__":
    main()
