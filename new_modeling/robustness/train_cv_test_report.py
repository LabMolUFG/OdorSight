"""
Train / CV / Test robustness report (new curation, 4201) for the GNN and the classical
baselines (RF/SVM/XGBoost, Morgan/ECFP4). Addresses the reviewer request to report training
and cross-validation statistics (not just the test set), to demonstrate robustness / absence
of overfitting.

For each model, on the SAME primary split (seed 42):
  - TRAIN : metrics of the fitted model on its own training set (in-sample);
  - CV    : 5-fold stratified CV on the training set — per-fold metrics, reported as mean +/- SD;
  - TEST  : metrics on the held-out external test set.
All threshold-dependent metrics use a fixed 0.5 cut-off (clean, consistent across the three
partitions); ROC-AUC and PR-AUC are threshold-independent.

GNN: uses the published hyperparameters (best_params.json); the saved primary model
(gnn_model_newcuration.pth) provides TRAIN/TEST, and 5 fold-models provide the CV.
Classical: uses the tuned hyperparameters saved in ml_baseline/results/best_params.json.

Outputs -> ./results/  (train_cv_test.csv + log.txt)
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
NM = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NM, "ml_baseline"))
import ml_common as M        # also puts uncertainty/common.py on the path
import common as C

OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
GNN_MODEL = os.path.join(NM, "newcuration_retrain", "gnn_model_newcuration.pth")
ML_BEST = os.path.join(NM, "ml_baseline", "results", "best_params.json")
THR = 0.5
METRICS = ["BACC", "MCC", "F1", "Precision", "Recall", "Specificity", "ROC_AUC", "PR_AUC"]
NFOLDS = 5
SEED = 42

logf = open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8")
def log(m=""):
    print(m); logf.write(str(m) + "\n"); logf.flush()

rows = []
def add(model, split, md, std=None):
    r = {"model": model, "split": split}
    for m in METRICS:
        r[m] = round(float(md[m]), 4)
        if std is not None:
            r[m + "_std"] = round(float(std[m]), 4)
    rows.append(r)

def agg(folds):
    mean = {m: float(np.mean([f[m] for f in folds])) for m in METRICS}
    std = {m: float(np.std([f[m] for f in folds], ddof=1)) for m in METRICS}
    return mean, std


def main():
    dev = C.get_device()
    log(f"Device: {dev} | threshold={THR} | {NFOLDS}-fold CV | new curation (4201)")
    pool = M.build_pool()
    y = pool["Outcome"].to_numpy().astype(int)
    tr_idx, te_idx = M.primary_split_indices(pool)
    log(f"Primary split (seed 42): train {len(tr_idx)} / test {len(te_idx)}")

    # ================= Classical baselines (Morgan/ECFP4) =================
    X = M.featurize(pool)[0]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]
    best = json.load(open(ML_BEST))
    for algo in ["rf", "svm", "xgb"]:
        params = best[algo]["params"]
        model = M.make_model(algo, params, ytr); model.fit(Xtr, ytr)
        add(algo.upper(), "train", C.metrics_at_threshold(ytr, model.predict_proba(Xtr)[:, 1], THR))
        skf = StratifiedKFold(NFOLDS, shuffle=True, random_state=SEED)
        folds = []
        for tri, vai in skf.split(Xtr, ytr):
            m = M.make_model(algo, params, ytr[tri]); m.fit(Xtr[tri], ytr[tri])
            folds.append(C.metrics_at_threshold(ytr[vai], m.predict_proba(Xtr[vai])[:, 1], THR))
        mean, std = agg(folds); add(algo.upper(), "cv", mean, std)
        add(algo.upper(), "test", C.metrics_at_threshold(yte, model.predict_proba(Xte)[:, 1], THR))
        log(f"  {algo.upper():4} train BACC={rows[-3]['BACC']:.3f} | "
            f"CV BACC={mean['BACC']:.3f}+/-{std['BACC']:.3f} | test BACC={rows[-1]['BACC']:.3f}")

    # ========================= GNN (GAT) =========================
    log("\nFeaturizing molecular graphs for the GNN...")
    def graphs(idx):
        out = []
        for s, l in zip(pool.loc[idx, "SMILES"], pool.loc[idx, "Outcome"]):
            d = C.smiles_to_data(s, label=int(l))
            if d is not None:
                out.append(d)
        return out
    train_g, test_g = graphs(tr_idx), graphs(te_idx)
    params_gnn = C.load_params(); ndf, edim = C.infer_dims(train_g)

    gnn = C.load_saved_model(params_gnn, ndf, edim, dev, GNN_MODEL)
    yt_tr, p_tr = C.predict_probs(gnn, train_g, dev)
    add("GNN", "train", C.metrics_at_threshold(yt_tr, p_tr, THR))
    yt_te, p_te = C.predict_probs(gnn, test_g, dev)

    lbl = np.array([d.y.item() for d in train_g])
    skf = StratifiedKFold(NFOLDS, shuffle=True, random_state=SEED)
    folds = []
    for k, (tri, vai) in enumerate(skf.split(np.zeros(len(train_g)), lbl)):
        log(f"  GNN CV fold {k+1}/{NFOLDS} (train {len(tri)} / val {len(vai)})...")
        m = C.train_model([train_g[i] for i in tri], params_gnn, ndf, edim, dev, epochs=80, seed=SEED + k)
        yv, pv = C.predict_probs(m, [train_g[i] for i in vai], dev)
        folds.append(C.metrics_at_threshold(yv, pv, THR))
    mean, std = agg(folds)
    add("GNN", "cv", mean, std)
    add("GNN", "test", C.metrics_at_threshold(yt_te, p_te, THR))
    log(f"  GNN  train BACC={rows[-3]['BACC']:.3f} | CV BACC={mean['BACC']:.3f}+/-{std['BACC']:.3f} "
        f"| test BACC={rows[-1]['BACC']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "train_cv_test.csv"), index=False)
    log("\n===== TRAIN / CV / TEST (threshold 0.5) =====")
    show = ["BACC", "MCC", "F1", "ROC_AUC", "PR_AUC"]
    log(f"{'model':5} {'split':6} " + " ".join(f"{m:>8}" for m in show))
    for r in rows:
        cv = r["split"] == "cv"
        log(f"{r['model']:5} {r['split']:6} " +
            " ".join((f"{r[m]:.3f}±{r[m+'_std']:.02f}" if cv else f"{r[m]:8.3f}") for m in show))
    log(f"\nSaved to {OUT}")
    logf.close()


if __name__ == "__main__":
    main()
