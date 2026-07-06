"""
common.py — Shared utilities for the uncertainty-quantification analysis
(reviewer response: bootstrap CIs + variance across repeated splits).

This module deliberately reuses the EXACT model definition and feature
pipeline from the original training script so that the saved weights
(gnn_best_model_cv5.pth) load and reproduce the published metrics.

Nothing here overwrites the original model, threshold, or datasets.
"""

import os
import sys
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    balanced_accuracy_score, matthews_corrcoef, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve,
)

# --- Make gnn_utils (in the parent Modeling/ dir) importable ---------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODELING = os.path.dirname(HERE)
DATA_DIR = os.path.join(MODELING, "DATA")
if MODELING not in sys.path:
    sys.path.insert(0, MODELING)

from gnn_utils import smiles_to_data  # noqa: E402
from rdkit import Chem               # noqa: E402
from rdkit import RDLogger           # noqa: E402
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Model definition — VERBATIM copy of optimize_gnn_cv5.py:GNN_Optimized
# (kept here so we do not need to import optuna; weights load with strict=True)
# ---------------------------------------------------------------------------
class GNN_Optimized(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes,
                 heads=1, edge_dim=None, num_layers=3, dropout=0.5, model_type='GCN'):
        super(GNN_Optimized, self).__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.model_type = model_type

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # --- Input layer ---
        if model_type == 'GAT':
            self.convs.append(GATv2Conv(num_node_features, hidden_channels, heads=heads, edge_dim=edge_dim, concat=False))
        else:
            self.convs.append(GCNConv(num_node_features, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))

        # --- Hidden layers ---
        for _ in range(num_layers - 2):
            if model_type == 'GAT':
                self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=edge_dim, concat=False))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

        # --- Final conv layer ---
        if model_type == 'GAT':
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=edge_dim, concat=False))
        else:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))

        # Final classifier
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None):
        for i, conv in enumerate(self.convs):
            if self.model_type == 'GAT':
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def df_to_data_list(df, smiles_col='SMILES', label_col='Outcome'):
    """Convert a dataframe to a list of (PyG Data) objects, keeping the SMILES
    that converted successfully (for alignment with external predictions)."""
    data_list, kept_smiles, kept_labels = [], [], []
    for s, l in zip(df[smiles_col], df[label_col]):
        d = smiles_to_data(s, label=int(l))
        if d is not None:
            data_list.append(d)
            kept_smiles.append(s)
            kept_labels.append(int(l))
    return data_list, kept_smiles, np.array(kept_labels, dtype=int)


def infer_dims(data_list):
    num_node_features = data_list[0].num_node_features
    edge_dim = 0
    for d in data_list:
        if d.edge_attr is not None and d.edge_attr.dim() == 2 and d.edge_attr.shape[1] > 0:
            edge_dim = d.edge_attr.shape[1]
            break
    return num_node_features, edge_dim


def build_model(params, num_node_features, edge_dim, device):
    return GNN_Optimized(
        num_node_features=num_node_features,
        hidden_channels=params['hidden_channels'],
        num_classes=2,
        heads=params['heads'],
        edge_dim=edge_dim,
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        model_type=params['model_type'],
    ).to(device)


def load_saved_model(params, num_node_features, edge_dim, device, model_path):
    model = build_model(params, num_node_features, edge_dim, device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def class_weights_for(data_list, device):
    labels = [d.y.item() for d in data_list]
    counts = np.bincount(labels, minlength=2)
    return torch.tensor([len(labels) / (2 * c) if c > 0 else 0.0 for c in counts],
                        dtype=torch.float).to(device)


def train_model(train_list, params, num_node_features, edge_dim, device,
                epochs=80, seed=42):
    """Train a fresh model with FIXED hyperparameters (no Optuna).
    Mirrors the final-retrain loop of optimize_gnn_cv5.py."""
    set_seed(seed)
    model = build_model(params, num_node_features, edge_dim, device)
    loader = DataLoader(train_list, batch_size=params['batch_size'], shuffle=True)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_for(train_list, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                                  weight_decay=params['weight_decay'])
    model.train()
    for _ in range(epochs):
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_probs(model, data_list, device, batch_size=64):
    """Return (y_true, prob_positive) for a list of PyG Data with .y set."""
    model.eval()
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    y_true, y_prob = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
        p = F.softmax(out, dim=1)[:, 1]
        y_prob.extend(p.cpu().numpy())
        y_true.extend(data.y.cpu().numpy())
    return np.array(y_true, dtype=int), np.array(y_prob, dtype=float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_from_pred(y_true, y_pred, y_prob=None):
    """Compute the full metric panel from hard predictions; AUCs need y_prob."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        'BACC': balanced_accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': (tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'F1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        # AUCs are undefined if only one class present in the (resampled) labels
        if len(np.unique(y_true)) == 2:
            out['ROC_AUC'] = roc_auc_score(y_true, y_prob)
            out['PR_AUC'] = average_precision_score(y_true, y_prob)
        else:
            out['ROC_AUC'] = np.nan
            out['PR_AUC'] = np.nan
    return out


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    return metrics_from_pred(y_true, y_pred, y_prob)


METRIC_ORDER = ['BACC', 'MCC', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC_AUC', 'PR_AUC']


# ---------------------------------------------------------------------------
# Threshold selection (HONEST: never uses the external test set)
# ---------------------------------------------------------------------------
def pick_threshold(y_true, y_prob, criterion='bacc'):
    """Choose an operating threshold on (training-derived) predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if criterion == 'gmean':
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        g = np.sqrt(tpr * (1.0 - fpr))
        return float(thr[int(np.argmax(g))])
    if criterion == 'youden':
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        return float(thr[int(np.argmax(j))])
    # default: maximize Balanced Accuracy over candidate cut-points
    cands = np.unique(y_prob)
    best_t, best_s = 0.5, -1.0
    for t in cands:
        s = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def stratified_boot_indices(y_true, rng):
    """Resample WITH replacement within each class (keeps the class ratio fixed,
    guarantees both classes are present in every resample)."""
    y = np.asarray(y_true).astype(int)
    idx = np.arange(len(y))
    parts = []
    for c in np.unique(y):
        ci = idx[y == c]
        parts.append(rng.choice(ci, size=len(ci), replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def percentile_ci(samples, alpha=0.05):
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    lo = np.percentile(s, 100 * alpha / 2)
    hi = np.percentile(s, 100 * (1 - alpha / 2))
    return float(np.mean(s)), float(np.median(s)), float(lo), float(hi), float(np.std(s, ddof=1))


def load_params(model_dir=MODELING):
    with open(os.path.join(model_dir, 'best_params.json')) as f:
        return json.load(f)
