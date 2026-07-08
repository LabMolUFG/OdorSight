"""
Classical ML baselines (RF / SVM / XGBoost) for odorant classification, following the SAME
scientific standards used for the GNN:
  - new curation (Curation/curated_dataset.csv, 4201);
  - benchmark-restricted stratified 90/10 split (test disjoint from Odorify);
  - hyperparameters tuned once via 5-fold CV with BACC (mirroring the GNN's Optuna protocol);
  - class-imbalance handling; feature scaling for SVM;
  - honest decision threshold from training data only (never the test set);
  - uncertainty via bootstrap CIs + K=15 repeated splits;
  - identical metric panel (BACC, MCC, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC).

Two feature families per algorithm:
  - ECFP4  : Morgan fingerprints, radius 2, 2048 bits;
  - DESC   : 12 core physicochemical descriptors (extends the GNN's 4 global descriptors).
"""
import os, sys
import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator as _fpg
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
NEW_MODELING = os.path.dirname(HERE)
REPO = os.path.dirname(NEW_MODELING)
sys.path.insert(0, os.path.join(NEW_MODELING, "uncertainty"))
import common as C   # metrics, bootstrap, threshold, split helpers (shared with the GNN)

CURATED = os.path.join(REPO, "Curation", "curated_dataset.csv")
BACKUP = os.path.join(NEW_MODELING, "uncertainty", "_submitted_version_backup")
GNN_PRIMARY_DATA = os.path.join(NEW_MODELING, "newcuration_retrain", "DATA")
GNN_MODEL = os.path.join(NEW_MODELING, "newcuration_retrain", "gnn_model_newcuration.pth")

TEST_FRACTION = 0.10
CORE_DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "LabuteASA", "MolMR", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "FractionCSP3", "NumAromaticRings", "RingCount", "NumHeteroatoms",
]
_DESC_FUNCS = {n: getattr(Descriptors, n) for n in CORE_DESCRIPTORS}
_MORGAN = _fpg.GetMorganGenerator(radius=2, fpSize=2048)


# --------------------------------------------------------------------------- features
def _mol(smiles):
    return Chem.MolFromSmiles(str(smiles))

def ecfp4(smiles):
    m = _mol(smiles)
    if m is None:
        return None
    arr = np.zeros((2048,), dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(_MORGAN.GetFingerprint(m), arr)
    return arr

def core_desc(smiles):
    m = _mol(smiles)
    if m is None:
        return None
    out = np.empty((len(CORE_DESCRIPTORS),), dtype=np.float32)
    for i, n in enumerate(CORE_DESCRIPTORS):
        try:
            out[i] = _DESC_FUNCS[n](m)
        except Exception:
            out[i] = np.nan
    return out

def featurize(df, feature, smiles_col="SMILES"):
    """Return (X, valid_mask) aligned to df rows for the requested feature family."""
    fn = ecfp4 if feature == "ecfp4" else core_desc
    rows, valid = [], []
    dim = 2048 if feature == "ecfp4" else len(CORE_DESCRIPTORS)
    for s in df[smiles_col]:
        v = fn(s)
        if v is None:
            rows.append(np.full((dim,), np.nan, dtype=np.float32)); valid.append(False)
        else:
            rows.append(v); valid.append(True)
    return np.vstack(rows), np.array(valid)


# --------------------------------------------------------------------------- pool / split
def inchikey(s):
    m = _mol(s)
    return Chem.MolToInchiKey(m) if m else None

def build_pool():
    """New-curation pool with reconstructed Odorify-exclusivity flags (same as GNN)."""
    old = pd.concat([pd.read_csv(os.path.join(BACKUP, "train_dataset.csv")),
                     pd.read_csv(os.path.join(BACKUP, "test_dataset.csv"))], ignore_index=True)
    flag = {k: bool(v) for k, v in zip(old["InChIKey"], old["no_overlap"])}
    df = pd.read_csv(CURATED).rename(columns={"Label": "Outcome", "final_smiles": "SMILES"})
    df["InChIKey"] = df["SMILES"].map(inchikey)
    df = df.dropna(subset=["InChIKey"]).reset_index(drop=True)
    df["no_overlap"] = df["InChIKey"].map(lambda k: flag.get(k, False))
    df["canon"] = df["SMILES"].map(C.canonicalize_smiles)
    df["_idx"] = np.arange(len(df))
    return df

def constrained_split(pool, seed):
    """Reproduce the benchmark-restricted stratified 90/10 split for a given seed."""
    from sklearn.model_selection import train_test_split
    excl = pool[pool["no_overlap"]]; rest = pool[~pool["no_overlap"]]
    test_prop = int(len(pool) * TEST_FRACTION) / len(excl)
    tr_excl, te = train_test_split(excl, test_size=test_prop, stratify=excl["Outcome"], random_state=seed)
    train = pd.concat([tr_excl, rest], ignore_index=True)
    train = train[~train["canon"].isin(set(te["canon"]))]
    return train["_idx"].to_numpy(), te["_idx"].to_numpy()

def primary_split_indices(pool):
    """Indices of the GNN's primary split (seed 42) matched by InChIKey — same test molecules."""
    tr = pd.read_csv(os.path.join(GNN_PRIMARY_DATA, "train_dataset.csv"))
    te = pd.read_csv(os.path.join(GNN_PRIMARY_DATA, "test_dataset.csv"))
    key2idx = {k: i for i, k in zip(pool["_idx"], pool["InChIKey"])}
    tr_idx = np.array([key2idx[k] for k in tr["InChIKey"] if k in key2idx])
    te_idx = np.array([key2idx[k] for k in te["InChIKey"] if k in key2idx])
    return tr_idx, te_idx


# --------------------------------------------------------------------------- models
def make_model(algo, params, y_train, probability=True):
    """Build a sklearn Pipeline with imputation (+scaling for SVM) and imbalance handling.
    probability=False skips SVM Platt scaling (much faster) — used during tuning."""
    steps = [("imp", SimpleImputer(strategy="median"))]
    if algo == "rf":
        clf = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=0, **params)
    elif algo == "svm":
        steps.append(("sc", StandardScaler()))
        clf = SVC(class_weight="balanced", probability=probability, random_state=0, **params)
    elif algo == "xgb":
        n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
        clf = XGBClassifier(
            scale_pos_weight=(n_neg / max(n_pos, 1)), n_jobs=-1, random_state=0,
            tree_method="hist", eval_metric="logloss", **params)
    else:
        raise ValueError(algo)
    steps.append(("clf", clf))
    return Pipeline(steps)

def _suggest(trial, algo):
    if algo == "rf":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]))
    if algo == "svm":
        return dict(
            C=trial.suggest_float("C", 1e-1, 1e2, log=True),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel="rbf")
    if algo == "xgb":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 700, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 6),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))

def tune(algo, X, y, n_trials, seed=42):
    """Tune HPs with 5-fold CV BACC (mirrors the GNN's Optuna + CV5 protocol)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    def objective(trial):
        params = _suggest(trial, algo)
        scores = []
        for tr, va in skf.split(X, y):
            model = make_model(algo, params, y[tr], probability=False)  # fast (no Platt)
            model.fit(X[tr], y[tr])
            scores.append(balanced_accuracy_score(y[va], model.predict(X[va])))
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def oof_threshold(algo, params, X, y, seed=42):
    """Honest threshold: 5-fold OOF proba on training data → maximize BACC."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(make_model(algo, params, y), X, y, cv=skf,
                            method="predict_proba", n_jobs=1)[:, 1]
    return C.pick_threshold(y, oof, "bacc")
