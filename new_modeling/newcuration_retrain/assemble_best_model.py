"""
Regenerate the BEST-BACC split/model from the K=15 repeated splits (new curation)
and save it, deterministically, into the deliverable folder.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
MODELING = os.path.dirname(HERE)
REPO = os.path.dirname(MODELING)
sys.path.insert(0, os.path.join(MODELING, "uncertainty"))
import common as C

TEST_FRACTION = 0.10
EPOCHS = 80
CURATED = os.path.join(REPO, "Curation", "curated_dataset.csv")
BACKUP = os.path.join(MODELING, "uncertainty", "_submitted_version_backup")
PER_SEED = os.path.join(HERE, "uncertainty_results", "repeated_splits_per_seed.csv")
OUT = os.path.join(MODELING, "reviewer_deliverable", "melhor_modelo")
os.makedirs(OUT, exist_ok=True)


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
    return df, data


def main():
    device = C.get_device(); params = C.load_params()
    ps = pd.read_csv(PER_SEED)
    best = ps.loc[ps["BACC"].idxmax()]
    seed = int(best["seed"])
    print(f"Best split by BACC: seed {seed} | recorded BACC={best['BACC']:.4f} MCC={best['MCC']:.4f} F1={best['F1']:.4f}")

    pool, data = build_pool()
    ndf, edim = C.infer_dims([d for d in data if d is not None])
    excl = pool[(pool["no_overlap"]) & (pool["valid"])]; rest = pool[(~pool["no_overlap"]) & (pool["valid"])]
    test_prop = int(len(pool) * TEST_FRACTION) / len(excl)
    tr_excl, te = train_test_split(excl, test_size=test_prop, stratify=excl["Outcome"], random_state=seed)
    train_df = pd.concat([tr_excl, rest], ignore_index=True)
    train_df = train_df[~train_df["canon"].isin(set(te["canon"]))]

    tr_idx = train_df["_idx"].to_numpy(); te_idx = te["_idx"].to_numpy()
    tr_lbl = pool.loc[tr_idx, "Outcome"].to_numpy()
    ti, vi = train_test_split(tr_idx, test_size=0.15, stratify=tr_lbl, random_state=seed)

    model = C.train_model([data[i] for i in ti], params, ndf, edim, device, epochs=EPOCHS, seed=seed)
    yv, pv = C.predict_probs(model, [data[i] for i in vi], device)
    thr = C.pick_threshold(yv, pv, "bacc")
    yt, pt = C.predict_probs(model, [data[i] for i in te_idx], device)
    m_thr = C.metrics_at_threshold(yt, pt, thr)
    m_05 = C.metrics_at_threshold(yt, pt, 0.5)
    print(f"Reproduced: BACC={m_thr['BACC']:.4f} MCC={m_thr['MCC']:.4f} F1={m_thr['F1']:.4f} (thr={thr:.4f})")

    # --- save split (full train + test) ---
    cols = ["Outcome", "SMILES", "InChI", "InChIKey", "no_overlap"]
    train_df[cols].to_csv(os.path.join(OUT, "train_dataset.csv"), index=False)
    te[cols].to_csv(os.path.join(OUT, "test_dataset.csv"), index=False)
    # --- save model + params + threshold + metrics ---
    torch.save(model.state_dict(), os.path.join(OUT, "gnn_model_best.pth"))
    import shutil
    shutil.copy(os.path.join(MODELING, "best_params.json"), os.path.join(OUT, "best_params.json"))
    with open(os.path.join(OUT, "model_threshold.txt"), "w") as f:
        f.write(str(thr))
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump({
            "selection": "highest BACC among K=15 repeated splits (new curation)",
            "seed": seed, "curation": "curated_dataset.csv (4201)",
            "n_train_full": int(len(train_df)), "n_train_used_for_model": int(len(ti)),
            "n_val_for_threshold": int(len(vi)), "n_test": int(len(te_idx)),
            "test_pos": int(yt.sum()), "test_neg": int((yt == 0).sum()),
            "threshold": float(thr),
            "metrics_at_threshold": m_thr, "metrics_at_0.5": m_05,
            "note": "Model trained on 85% of the split's training set; threshold chosen on the "
                    "remaining 15% (never the test set). train_dataset.csv is the full split train.",
        }, f, indent=2)
    print(f"Saved best model + split to: {OUT}")


if __name__ == "__main__":
    main()
