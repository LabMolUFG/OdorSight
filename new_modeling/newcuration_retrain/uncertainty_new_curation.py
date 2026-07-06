"""
Uncertainty quantification for the NEW-curation model (reviewer: CIs / repeated splits).
Mirrors the submitted-version analysis, applied to Curation/curated_dataset.csv (4201).

  Approach A — bootstrap 95% CIs on the primary external test set, using the retrained
               model (gnn_model_newcuration.pth) at an HONEST threshold derived from the
               training set only (5-fold OOF CV), plus 0.5 as reference.
  Approach B — variance across K constrained, stratified splits (test drawn only from
               Odorify-exclusive molecules), retraining with the published hyperparameters.

NOTE: the paired Odorify comparison is NOT reproduced here — Odorify's per-molecule
predictions exist only for the OLD 421-molecule test set; a new test set has different
molecules and would require running the external Odorify tool.

Outputs -> ./uncertainty_results/
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
MODELING = os.path.dirname(HERE)
REPO = os.path.dirname(MODELING)
sys.path.insert(0, os.path.join(MODELING, "uncertainty"))
import common as C

TEST_FRACTION = 0.10
CURATED = os.path.join(REPO, "Curation", "curated_dataset.csv")
BACKUP  = os.path.join(MODELING, "uncertainty", "_submitted_version_backup")
PRIMARY_DATA = os.path.join(HERE, "DATA")
MODEL_PATH = os.path.join(HERE, "gnn_model_newcuration.pth")
OUT = os.path.join(HERE, "uncertainty_results"); os.makedirs(OUT, exist_ok=True)

logf = open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8")
def log(m=""):
    print(m); logf.write(str(m) + "\n"); logf.flush()


def inchikey(s):
    m = C.Chem.MolFromSmiles(str(s)); return C.Chem.MolToInchiKey(m) if m else None


def build_pool():
    old = pd.concat([pd.read_csv(os.path.join(BACKUP, "train_dataset.csv")),
                     pd.read_csv(os.path.join(BACKUP, "test_dataset.csv"))], ignore_index=True)
    flag = {k: bool(v) for k, v in zip(old["InChIKey"], old["no_overlap"])}
    df = pd.read_csv(CURATED).rename(columns={"Label": "Outcome", "final_smiles": "SMILES"})
    df["InChIKey"] = df["SMILES"].map(inchikey)
    df = df.dropna(subset=["InChIKey"]).reset_index(drop=True)
    df["no_overlap"] = df["InChIKey"].map(lambda k: flag.get(k, False))
    data, canon = [], []
    for s, l in zip(df["SMILES"], df["Outcome"]):
        data.append(C.smiles_to_data(s, label=int(l))); canon.append(C.canonicalize_smiles(s))
    df["valid"] = [d is not None for d in data]; df["canon"] = canon; df["_idx"] = np.arange(len(df))
    log(f"Pool: {len(df)} cpds | Odorify-exclusive {int(df['no_overlap'].sum())} | featurized {int(df['valid'].sum())}")
    return df, data


def cv5_threshold(train_list, params, ndf, edim, device, epochs, seed):
    labels = np.array([d.y.item() for d in train_list])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    yt, pt = [], []
    for k, (tr, va) in enumerate(skf.split(np.arange(len(train_list)), labels)):
        log(f"    [cv5] fold {k+1}/5")
        m = C.train_model([train_list[i] for i in tr], params, ndf, edim, device, epochs=epochs, seed=seed + k)
        y, p = C.predict_probs(m, [train_list[i] for i in va], device)
        yt.extend(y); pt.extend(p)
    return C.pick_threshold(np.array(yt), np.array(pt), "bacc")


def approach_A(pool, data, ndf, edim, device, params, args):
    log("\n===== Approach A: bootstrap CIs on the retrained primary model =====")
    ptr = pd.read_csv(os.path.join(PRIMARY_DATA, "train_dataset.csv"))
    pte = pd.read_csv(os.path.join(PRIMARY_DATA, "test_dataset.csv"))
    idx = {k: i for i, k in zip(pool["_idx"], pool["InChIKey"])}
    tr_list = [data[idx[k]] for k in ptr["InChIKey"] if k in idx and data[idx[k]] is not None]
    te_list = [data[idx[k]] for k in pte["InChIKey"] if k in idx and data[idx[k]] is not None]
    model = C.load_saved_model(params, ndf, edim, device, MODEL_PATH)
    y_true, prob = C.predict_probs(model, te_list, device)
    log(f"Primary test N={len(y_true)} | pos={int(y_true.sum())} neg={int((y_true==0).sum())}")

    log("Deriving honest threshold (5-fold OOF on the primary training set)...")
    t_honest = cv5_threshold(tr_list, params, ndf, edim, device, args.epochs, args.seed)
    t_opt = C.pick_threshold(y_true, prob, "gmean")
    thresholds = {f"honest_cv5={t_honest:.3f}": t_honest, "ref_0.5": 0.5, f"ref_opt_test={t_opt:.3f}": t_opt}
    log(f"Honest threshold (train CV) = {t_honest:.4f}")

    rng = np.random.default_rng(args.seed)
    boot = {lab: {m: [] for m in C.METRIC_ORDER} for lab in thresholds}
    for _ in range(args.n_boot):
        bi = C.stratified_boot_indices(y_true, rng)
        for lab, thr in thresholds.items():
            mm = C.metrics_at_threshold(y_true[bi], prob[bi], thr)
            for m in C.METRIC_ORDER: boot[lab][m].append(mm[m])

    rows = []
    for lab, thr in thresholds.items():
        pt = C.metrics_at_threshold(y_true, prob, thr)
        for m in C.METRIC_ORDER:
            mean, med, lo, hi, sd = C.percentile_ci(boot[lab][m])
            rows.append({"threshold": lab, "metric": m, "point": pt[m],
                         "ci95_low": lo, "ci95_high": hi, "boot_std": sd})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bootstrap_summary.csv"), index=False)
    prim = f"honest_cv5={t_honest:.3f}"
    log(f"\n{'metric':12} {'point':>8} {'95% CI':>22}  (honest threshold)")
    for m in C.METRIC_ORDER:
        r = [x for x in rows if x["threshold"] == prim and x["metric"] == m][0]
        log(f"{m:12} {r['point']:8.4f}   [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]")
    return t_honest


def approach_B(pool, data, ndf, edim, device, params, args):
    log("\n===== Approach B: variance across repeated constrained splits =====")
    per_seed = os.path.join(OUT, "repeated_splits_per_seed.csv")
    excl = pool[(pool["no_overlap"]) & (pool["valid"])]; rest = pool[(~pool["no_overlap"]) & (pool["valid"])]
    test_size = int(len(pool) * TEST_FRACTION); test_prop = test_size / len(excl)
    for k, seed in enumerate(range(args.start_seed, args.start_seed + args.seeds)):
        tr_excl, te = train_test_split(excl, test_size=test_prop, stratify=excl["Outcome"], random_state=seed)
        train_df = pd.concat([tr_excl, rest], ignore_index=True)
        tset = set(te["canon"]); train_df = train_df[~train_df["canon"].isin(tset)]
        tr_idx = train_df["_idx"].to_numpy(); te_idx = te["_idx"].to_numpy()
        tr_lbl = pool.loc[tr_idx, "Outcome"].to_numpy()
        ti, vi = train_test_split(tr_idx, test_size=0.15, stratify=tr_lbl, random_state=seed)
        model = C.train_model([data[i] for i in ti], params, ndf, edim, device, epochs=args.epochs, seed=seed)
        yv, pv = C.predict_probs(model, [data[i] for i in vi], device)
        thr = C.pick_threshold(yv, pv, "bacc")
        yt, pt = C.predict_probs(model, [data[i] for i in te_idx], device)
        row = {"seed": seed, "threshold": thr, "n_train": len(ti), "n_test": len(te_idx),
               **{m: C.metrics_at_threshold(yt, pt, thr)[m] for m in C.METRIC_ORDER},
               **{f"{m}@0.5": C.metrics_at_threshold(yt, pt, 0.5)[m] for m in ("BACC", "MCC", "F1")}}
        pd.DataFrame([row]).to_csv(per_seed, mode="a", header=not os.path.exists(per_seed), index=False)
        log(f"seed {seed} ({k+1}/{args.seeds}): BACC={row['BACC']:.4f} MCC={row['MCC']:.4f} F1={row['F1']:.4f} thr={thr:.3f}")

    df = pd.read_csv(per_seed)
    summ = []
    log(f"\n{'metric':12} {'mean':>8} {'std':>8} {'95% range':>22}")
    for m in C.METRIC_ORDER:
        mean, med, lo, hi, sd = C.percentile_ci(df[m].to_numpy())
        summ.append({"metric": m, "mean": mean, "std": sd, "ci95_low": lo, "ci95_high": hi,
                     "min": float(df[m].min()), "max": float(df[m].max())})
        log(f"{m:12} {mean:8.4f} {sd:8.4f}   [{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "repeated_splits_summary.csv"), index=False)
    show = [m for m in ["BACC", "MCC", "F1", "Precision", "Recall", "ROC_AUC", "PR_AUC"] if m in df.columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([df[m] for m in show], labels=show, showmeans=True)
    ax.set_ylim(0, 1.02); ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"New curation — external-test metrics across {len(df)} splits")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "repeated_splits_distribution.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=["A", "B"], default=None)
    args = ap.parse_args()
    device = C.get_device(); params = C.load_params()
    log(f"Device: {device} | params: {params} | B={args.n_boot} seeds={args.seeds}")
    pool, data = build_pool()
    ndf, edim = C.infer_dims([d for d in data if d is not None])
    if args.only != "B":
        approach_A(pool, data, ndf, edim, device, params, args)
    if args.only != "A":
        approach_B(pool, data, ndf, edim, device, params, args)
    log(f"\nSaved to {OUT}")
    logf.close()


if __name__ == "__main__":
    main()
