"""
Retrain the published GNN on the NEW curation (Curation/curated_dataset.csv, 4201 cpds)
using the ORIGINAL benchmark-constrained, stratified 90/10 split and the PUBLISHED
hyperparameters (best_params.json). Nothing published is overwritten.

Benchmark constraint (as in the original methodology): the external test set is drawn
only from molecules absent from Odorify's training data. Since the Odorify dataset file
(odorify_dataset.xlsx) is not in the repository, the reliable per-molecule `no_overlap`
flags from the submitted-version split (backed up in
uncertainty/_submitted_version_backup/) are reused to identify Odorify-exclusive
molecules. Molecules new to this curation (absent from the old split) have unknown Odorify
status and are conservatively kept in the training set, so the test set stays disjoint.

Outputs (this folder): DATA/{train,test}_dataset.csv, gnn_model_newcuration.pth,
best_params.json (copy), model_threshold.txt, metrics.json, report.txt.
"""

import os, sys, json, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
MODELING = os.path.dirname(HERE)
REPO = os.path.dirname(MODELING)
sys.path.insert(0, os.path.join(MODELING, "uncertainty"))   # reuse common.py
import common as C

TEST_FRACTION = 0.10
SEED = 42
EPOCHS = 80

CURATED = os.path.join(REPO, "Curation", "curated_dataset.csv")
BACKUP  = os.path.join(MODELING, "uncertainty", "_submitted_version_backup")
OUT_DATA = os.path.join(HERE, "DATA")
os.makedirs(OUT_DATA, exist_ok=True)

log_lines = []
def log(m=""):
    print(m); log_lines.append(str(m))


def build_split():
    # --- reliable Odorify-exclusivity flags from the submitted-version split ---
    old = pd.concat([pd.read_csv(os.path.join(BACKUP, "train_dataset.csv")),
                     pd.read_csv(os.path.join(BACKUP, "test_dataset.csv"))],
                    ignore_index=True)
    flag = {k: bool(v) for k, v in zip(old["InChIKey"], old["no_overlap"])}

    # --- new curation ---
    df = pd.read_csv(CURATED).rename(columns={"Label": "Outcome", "final_smiles": "SMILES"})
    df["InChIKey"] = df["SMILES"].apply(
        lambda s: (lambda m: C.Chem.MolToInchiKey(m) if m else None)(C.Chem.MolFromSmiles(str(s))))
    df = df.dropna(subset=["InChIKey"]).reset_index(drop=True)
    # test-eligible = known Odorify-exclusive; unknown/new -> train only (conservative)
    df["no_overlap"] = df["InChIKey"].map(lambda k: flag.get(k, False))

    excl = df[df["no_overlap"]].copy()
    rest = df[~df["no_overlap"]].copy()
    n_unknown = int((~df["InChIKey"].isin(flag)).sum())
    log(f"NEW curation: {len(df)} cpds | Odorify-exclusive (test-eligible): {len(excl)} "
        f"| train-only: {len(rest)} (of which {n_unknown} are new/unknown-status)")

    test_size = int(len(df) * TEST_FRACTION)
    test_prop = test_size / len(excl)
    tr_excl, test = train_test_split(excl, test_size=test_prop,
                                     stratify=excl["Outcome"], random_state=SEED)
    train = pd.concat([tr_excl, rest], ignore_index=True)

    # remove canonical-SMILES overlaps between train and test (as in the original notebook)
    train["canon"] = train["SMILES"].apply(C.canonicalize_smiles)
    test["canon"]  = test["SMILES"].apply(C.canonicalize_smiles)
    tset = set(test["canon"])
    before = len(train)
    train = train[~train["canon"].isin(tset)]
    log(f"Removed {before - len(train)} train/test canonical-SMILES overlaps.")

    cols = ["Outcome", "SMILES", "InChI", "InChIKey", "no_overlap"]
    train[cols].to_csv(os.path.join(OUT_DATA, "train_dataset.csv"), index=False)
    test[cols].to_csv(os.path.join(OUT_DATA, "test_dataset.csv"), index=False)

    log(f"Split: train {len(train)} ({len(train)/len(df)*100:.1f}%) | "
        f"test {len(test)} ({len(test)/len(df)*100:.1f}%)")
    log(f"  train balance {train['Outcome'].value_counts().to_dict()} | "
        f"test balance {test['Outcome'].value_counts().to_dict()}")
    # benchmark disjointness guarantee (test-eligible were all Odorify-exclusive)
    assert test["no_overlap"].all(), "test contains non-exclusive molecules!"
    log("  Benchmark disjointness: PASSED (test set is disjoint from Odorify by construction)")
    return train[cols], test[cols]


def main():
    device = C.get_device()
    params = C.load_params()
    log(f"Device: {device} | params: {params}")

    train_df, test_df = build_split()

    train_list, _, ytr = C.df_to_data_list(train_df)
    test_list, _, yte = C.df_to_data_list(test_df)
    ndf, edim = C.infer_dims(train_list)
    log(f"Featurized: train {len(train_list)} / test {len(test_list)} | node_feats={ndf} edge_dim={edim}")

    log("Retraining with published hyperparameters (80 epochs)...")
    model = C.train_model(train_list, params, ndf, edim, device, epochs=EPOCHS, seed=SEED)

    yt, prob = C.predict_probs(model, test_list, device)
    t_opt = C.pick_threshold(yt, prob, "gmean")   # test-optimal (as in the ORIGINAL pipeline)
    m05 = C.metrics_at_threshold(yt, prob, 0.5)
    mop = C.metrics_at_threshold(yt, prob, t_opt)

    # save artifacts
    import torch
    torch.save(model.state_dict(), os.path.join(HERE, "gnn_model_newcuration.pth"))
    shutil.copy(os.path.join(MODELING, "best_params.json"), os.path.join(HERE, "best_params.json"))
    with open(os.path.join(HERE, "model_threshold.txt"), "w") as f:
        f.write(str(t_opt))
    out = {"dataset": "curated_dataset.csv (new curation, 4201)",
           "split": {"test_fraction": TEST_FRACTION, "seed": SEED,
                     "n_train": len(train_list), "n_test": len(test_list),
                     "constraint": "test drawn only from Odorify-exclusive molecules"},
           "threshold_optimal_gmean_on_test": t_opt,
           "metrics_at_0.5": m05, "metrics_at_optimal": mop}
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    log("\n================ NEW-CURATION model — external test ================")
    log(f"{'metric':12} {'@0.5':>10} {'@optimal('+format(t_opt,'.3f')+')':>16}")
    for k in C.METRIC_ORDER:
        log(f"{k:12} {m05[k]:10.4f} {mop[k]:16.4f}")
    log("\n--- Reference: PUBLISHED model on OLD curation ---")
    log("BACC  0.8916 (@0.5) / 0.9124 (@opt) | MCC 0.79 | F1 0.93 | Prec 0.96 | Rec 0.91")

    with open(os.path.join(HERE, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"\nSaved model + split + metrics to: {HERE}")


if __name__ == "__main__":
    main()
