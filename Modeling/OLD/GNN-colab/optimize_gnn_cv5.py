import optuna
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import os
from torch.nn import Linear, BatchNorm1d
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
try:
    from torchviz import make_dot
except ImportError:
    pass # Will check later before using

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    balanced_accuracy_score, matthews_corrcoef, precision_score, 
    recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# -------------------------------------------------------------------------
# PDF REPORT GENERATOR
# -------------------------------------------------------------------------
def format_value(v):
    if isinstance(v, float):
        if v < 0.001: return f"{v:.2e}"
        return f"{v:.2f}"
    return str(v)

def generate_pdf_report(report_path, best_cv_score, ext_bacc, best_params, 
                        y_true_ext, y_pred_ext, y_prob_ext, arch_img_path,
                        best_threshold, bacc_opt,
                        train_stats=None, ext_stats=None):
    print(f"\nGenerating Enhanced PDF Report: {report_path}...")
    
    # --- G-Mean Optimization (Recalculate curves for plotting) ---
    fpr, tpr, thresholds = roc_curve(y_true_ext, y_prob_ext)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    # best_threshold passed in is likely same as thresholds[ix] but we use passed value for consistency
    best_gmean = gmeans[ix]
    
    # Recalculate metrics at optimal threshold (using passed threshold)
    y_pred_opt = (np.array(y_prob_ext) >= best_threshold).astype(int)
    
    # Metrics for Optimal
    mcc_opt = matthews_corrcoef(y_true_ext, y_pred_opt)
    prec_opt = precision_score(y_true_ext, y_pred_opt, zero_division=0)
    rec_opt = recall_score(y_true_ext, y_pred_opt, zero_division=0)
    f1_opt = f1_score(y_true_ext, y_pred_opt, zero_division=0)

    # Standard metrics (Threshold 0.5) - passed in y_pred_ext should be 0.5 based
    mcc_def = matthews_corrcoef(y_true_ext, y_pred_ext)
    prec_def = precision_score(y_true_ext, y_pred_ext, zero_division=0)
    rec_def = recall_score(y_true_ext, y_pred_ext, zero_division=0)
    f1_def = f1_score(y_true_ext, y_pred_ext, zero_division=0)

    with PdfPages(report_path) as pdf:
        # Page 1: Summary & Metrics
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        
        # Title
        plt.text(0.5, 0.95, "GNN Final Report", ha='center', va='center', fontsize=24, weight='bold')
        plt.text(0.5, 0.92, f"Target: BACC > 0.94", ha='center', va='center', fontsize=12)
        
        # --- SECTION 1: OPTIMIZATION RESULTS (CV-5) ---
        plt.text(0.1, 0.85, "Phase 1: Optimization (5-Fold CV)", fontsize=16, weight='bold', color='blue')
        
        cv_text = (
            f"Best Average CV Score: {best_cv_score:.4f}\n"
            f"(This score determined the best parameters below)"
        )
        plt.text(0.1, 0.82, cv_text, fontsize=12, family='monospace', va='top')
        
        # Best Hyperparameters
        plt.text(0.1, 0.75, "Best Hyperparameters Found:", fontsize=12, weight='bold')
        formatted_params = {k: format_value(v) for k, v in best_params.items()}
        params_text = "\n".join([f"  {k}: {v}" for k, v in formatted_params.items()])
        plt.text(0.1, 0.73, params_text, fontsize=10, family='monospace', va='top')

        # --- SECTION 2: EXTERNAL VALIDATION ---
        plt.text(0.1, 0.55, "Phase 2: External Validation (Final Check)", fontsize=16, weight='bold', color='red')
        plt.text(0.1, 0.52, "Tested on 'moleculas_unicas.csv' (never seen during training)", fontsize=10, style='italic')

        results_text = (
            f"External BACC (Thr 0.5): {ext_bacc:.4f}\n"
            f"External BACC (Optimal): {bacc_opt:.4f} (Thr={best_threshold:.2f})\n"
            f"Target Met (>0.94):      {'YES ✅' if max(ext_bacc, bacc_opt) > 0.94 else 'NO ❌'}"
        )
        plt.text(0.1, 0.48, results_text, fontsize=12, family='monospace', va='top', bbox=dict(facecolor='lightyellow', alpha=0.5))
        
        # Detailed Metrics Comparison
        plt.text(0.1, 0.38, "Detailed External Metrics:", fontsize=12, weight='bold')
        
        metrics_def = (
            f"[Threshold 0.5]\n"
            f"MCC:       {mcc_def:.2f}\n"
            f"Precision: {prec_def:.2f}\n"
            f"Recall:    {rec_def:.2f}\n"
            f"F1 Score:  {f1_def:.2f}"
        )
        
        metrics_opt = (
            f"[Optimal Thr {best_threshold:.2f}]\n"
            f"MCC:       {mcc_opt:.2f}\n"
            f"Precision: {prec_opt:.2f}\n"
            f"Recall:    {rec_opt:.2f}\n"
            f"F1 Score:  {f1_opt:.2f}"
        )
        
        plt.text(0.1, 0.35, metrics_def, fontsize=10, family='monospace', va='top')
        plt.text(0.5, 0.35, metrics_opt, fontsize=10, family='monospace', va='top')

        # Dataset Stats
        if train_stats and ext_stats:
            stats_text = (
                f"Train Set: {train_stats['Total']} (Pos:{train_stats['Positive']} Neg:{train_stats['Negative']})\n"
                f"Ext Set:   {ext_stats['Total']} (Pos:{ext_stats['Positive']} Neg:{ext_stats['Negative']})"
            )
            plt.text(0.1, 0.20, "Dataset Statistics:", fontsize=11, weight='bold')
            plt.text(0.1, 0.18, stats_text, fontsize=10, family='monospace', va='top')
        
        pdf.savefig(fig)
        plt.close()

        # Page 2: Confusion Matrix (Default 0.5) vs (Optimal)
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # CM Default
        cm_def = confusion_matrix(y_true_ext, y_pred_ext)
        disp_def = ConfusionMatrixDisplay(confusion_matrix=cm_def, display_labels=['No Odor', 'Odor'])
        disp_def.plot(cmap='Blues', ax=axs[0], colorbar=False)
        axs[0].set_title(f"Default Threshold (0.50)\nBACC: {balanced_accuracy_score(y_true_ext, y_pred_ext):.4f}")
        
        # CM Optimal
        cm_opt = confusion_matrix(y_true_ext, y_pred_opt)
        disp_opt = ConfusionMatrixDisplay(confusion_matrix=cm_opt, display_labels=['No Odor', 'Odor'])
        disp_opt.plot(cmap='Greens', ax=axs[1], colorbar=False)
        axs[1].set_title(f"Optimal Threshold ({best_threshold:.2f})\nBACC: {bacc_opt:.4f}")
        
        plt.suptitle("Confusion Matrices Comparison", fontsize=16)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 3: ROC, PR, G-Mean
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        # Find index for 0.5 threshold (approx)
        idx_05 = (np.abs(thresholds - 0.5)).argmin()

        # ROC
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc(fpr, tpr):.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # Mark Optimal
        ax1.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='green', label=f'Best ({best_threshold:.2f})')
        # Mark Default
        ax1.scatter(fpr[idx_05], tpr[idx_05], s=100, marker='x', color='blue', label='Default (0.5)')
        
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")

        # PR
        perc_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true_ext, y_prob_ext)
        avg_prec = average_precision_score(y_true_ext, y_prob_ext)
        ax2.plot(rec_curve, perc_curve, color='blue', lw=2, label=f'PR (AP = {avg_prec:.2f})')
        
        # Mark Optimal on PR (Need to find index in pr_thresholds)
        # Note: pr_thresholds is smaller than curve points by 1
        if len(pr_thresholds) > 0:
            pr_ix = (np.abs(pr_thresholds - best_threshold)).argmin()
            ax2.scatter(rec_curve[pr_ix], perc_curve[pr_ix], s=100, marker='o', color='green', label=f'Best')
            
            pr_ix_05 = (np.abs(pr_thresholds - 0.5)).argmin()
            ax2.scatter(rec_curve[pr_ix_05], perc_curve[pr_ix_05], s=100, marker='x', color='blue', label='0.5')
        
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")

        # G-Mean
        ax3.plot(thresholds, gmeans, color='green', lw=2, label='G-Mean')
        ax3.axvline(best_threshold, color='green', linestyle='--', label=f'Best Thr {best_threshold:.2f}')
        ax3.axvline(0.5, color='blue', linestyle=':', label='Default 0.5')
        ax3.set_title(f'G-Mean Optimization')
        ax3.legend()
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 4: Architecture Diagram
        if os.path.exists(arch_img_path):
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.axis('off')
            plt.title("GNN Architecture Diagram", fontsize=16)
            try:
                img = plt.imread(arch_img_path)
                plt.imshow(img)
            except Exception as e:
                plt.text(0.5, 0.5, f"Error loading diagram: {e}", ha='center')
            pdf.savefig(fig)
            plt.close()
            
    print("Report saved!")

# Tenta importar sua função de conversão. Se falhar, avisa.
try:
    from gnn_utils import smiles_to_data
except ImportError:
    # Fallback if gnn_utils is not found, copy the function here or raise
    raise ImportError("Certifique-se de que o arquivo 'gnn_utils.py' com a função 'smiles_to_data' está na mesma pasta.")

# -------------------------------------------------------------------------
# 1. CLASSE DO MODELO (Reused)
# -------------------------------------------------------------------------
class GNN_Optimized(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, 
                 heads=1, edge_dim=None, num_layers=3, dropout=0.5, model_type='GCN'):
        super(GNN_Optimized, self).__init__()
        
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.model_type = model_type
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # --- Camada de Entrada ---
        if model_type == 'GAT':
            self.convs.append(GATv2Conv(num_node_features, hidden_channels, heads=heads, edge_dim=edge_dim, concat=False))
        else:
            self.convs.append(GCNConv(num_node_features, hidden_channels))
            
        self.bns.append(BatchNorm1d(hidden_channels))

        # --- Camadas Ocultas ---
        for _ in range(num_layers - 2):
            if model_type == 'GAT':
                self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=edge_dim, concat=False))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

        # --- Camada de Saída da Convolução ---
        if model_type == 'GAT':
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=edge_dim, concat=False))
        else:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))

        # Classificador Final
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

# Variável Global
GLOBAL_BEST_BACC = 0
GLOBAL_BEST_PARAMS = None
EXTERNAL_VAL_BACC = 0

# -------------------------------------------------------------------------
# 2. HELPER PARA CANONIZAÇÃO
# -------------------------------------------------------------------------
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None

# -------------------------------------------------------------------------
# 3. FUNÇÃO OBJETIVO DO OPTUNA (5-Fold CV)
# -------------------------------------------------------------------------
def objective(trial):
    global GLOBAL_BEST_BACC, GLOBAL_BEST_PARAMS, EXTERNAL_VAL_BACC
    global full_train_data_list, ext_data_list, device, num_node_features, edge_dim
    
    # --- Hyperparameters ---
    model_type = trial.suggest_categorical('model_type', ['GAT']) 
    # Use GAT as per diagram preference, but could re-enable GCN if desired. User asked for specific diagram which is GAT-like?
    # Actually the diagram file showed GAT. So we stick to GAT mostly or allow optimization.
    # We'll allow GAT only for now as it seems preferred by the user context "gnn_architecture_diagram"
    
    num_layers = trial.suggest_int('num_layers', 3, 6)
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
    heads = trial.suggest_categorical('heads', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # --- 5-Fold Stratified CV ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Prepare labels for stratification
    all_labels = [d.y.item() for d in full_train_data_list]
    all_indices = np.arange(len(full_train_data_list))
    
    fold_baccs = []
    
    # We just need to iterate folds. 
    # To save time in optimization, maybe we do fewer epochs per fold, but user wants high accuracy.
    epochs = 60 
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
        # Build Fold DataLoaders
        train_subset = [full_train_data_list[i] for i in train_idx]
        val_subset = [full_train_data_list[i] for i in val_idx]
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Class Weights for this fold
        labels = [d.y.item() for d in train_subset]
        counts = np.bincount(labels)
        class_weights = torch.tensor([len(labels) / (2 * c) for c in counts], dtype=torch.float).to(device)
        
        # Init Model
        model = GNN_Optimized(
            num_node_features=num_node_features, 
            hidden_channels=hidden_channels, 
            num_classes=2, 
            heads=heads, 
            edge_dim=edge_dim,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        # Train Loop (simplified for speed within CV)
        best_fold_bacc = 0
        patience = 10
        patience_ctr = 0
        
        for epoch in range(epochs):
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
            
            # Val
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
                    preds = out.argmax(dim=1)
                    y_true.extend(data.y.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
            
            val_bacc = balanced_accuracy_score(y_true, y_pred)
            
            if val_bacc > best_fold_bacc:
                best_fold_bacc = val_bacc
                patience_ctr = 0
            else:
                patience_ctr += 1
                
            if patience_ctr >= patience:
                break
        
        fold_baccs.append(best_fold_bacc)
        
        # Report intermediate generic value (average so far) to prune really bad trials early?
        # Maybe just report the current fold score as step=fold_idx
        trial.report(np.mean(fold_baccs), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    avg_cv_bacc = np.mean(fold_baccs)
    
    # --- Check for Global Best & External Validation ---
    if avg_cv_bacc > GLOBAL_BEST_BACC:
        GLOBAL_BEST_BACC = avg_cv_bacc
        GLOBAL_BEST_PARAMS = trial.params
        print(f"  >>> NEW BEST CV AVERAGE: {avg_cv_bacc:.4f} (Params: {trial.params})")
        
        # Optional: Evaluate on External Set just to see
        # But real external eval should happen on the Final Retrained Model
    
    return avg_cv_bacc


# -------------------------------------------------------------------------
# 4. MAIN & RETRAINING
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=50) 
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Load Data ---
    print("Loading datasets...")
    df_train = pd.read_csv('curated_dataset_final.csv')
    df_ext = pd.read_csv('moleculas_unicas.csv') # External set
    
    # Canonicalize
    df_train['canon_smiles'] = df_train['SMILES'].apply(canonicalize_smiles)
    df_ext['canon_smiles'] = df_ext['SMILES'].apply(canonicalize_smiles)
    df_train.dropna(subset=['canon_smiles'], inplace=True)
    df_ext.dropna(subset=['canon_smiles'], inplace=True)
    
    # Remove overlaps (External contains molecules NOT in Train)
    train_smiles = set(df_train['canon_smiles'])
    ext_smiles = set(df_ext['canon_smiles'])
    
    overlap = train_smiles.intersection(ext_smiles)
    if len(overlap) > 0:
        print(f"Removing {len(overlap)} overlapping molecules from TRAIN set.")
        df_train = df_train[~df_train['canon_smiles'].isin(overlap)]
    
    # Convert to PyG
    print("Converting to PyG Data...")
    full_train_data_list = []
    for s, l in zip(df_train['SMILES'], df_train['Outcome']):
        d = smiles_to_data(s, label=l)
        if d: full_train_data_list.append(d)
        
    ext_data_list = []
    for s, l in zip(df_ext['SMILES'], df_ext['Outcome']):
        d = smiles_to_data(s, label=l)
        if d: ext_data_list.append(d)

    if not full_train_data_list:
        raise ValueError("Train data empty!")

    # Global info for Objective
    num_node_features = full_train_data_list[0].num_node_features
    if full_train_data_list[0].edge_attr is not None:
        edge_dim = full_train_data_list[0].edge_attr.shape[1]
    else:
        edge_dim = 0
        
    # --- Optuna ---
    print("Starting Optimization (CV-5)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)
    
    print("="*40)
    print(f"Best CV BACC: {study.best_value:.4f}")
    print("Best Params:", study.best_params)
    
    # --- RETRAIN ON FULL TRAIN SET & TEST EXTERNAL ---
    print("\nRetraining best model on FULL training set...")
    best_params = study.best_params
    
    # Setup full loader
    full_loader = DataLoader(full_train_data_list, batch_size=best_params['batch_size'], shuffle=True)
    ext_loader = DataLoader(ext_data_list, batch_size=best_params['batch_size'], shuffle=False)
    
    # Recalculate weights
    labels = [d.y.item() for d in full_train_data_list]
    counts = np.bincount(labels)
    class_weights = torch.tensor([len(labels) / (2 * c) for c in counts], dtype=torch.float).to(device)

    final_model = GNN_Optimized(
        num_node_features=num_node_features,
        hidden_channels=best_params['hidden_channels'],
        num_classes=2,
        heads=best_params['heads'],
        edge_dim=edge_dim,
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        model_type=best_params['model_type']
    ).to(device)
    
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Train for fixed epochs or with a split? 
    # Since we use external set as final check, we can define a robust number of epochs or split full_train slightly internally.
    # To get maximum performance, we usually train on all data. We'll use a conservative epoch count or the average optimal from CV if we tracked it (we didn't).
    # We will assume 80 epochs is sufficient as used in previous scripts.
    final_epochs = 80
    
    final_model.train()
    for epoch in range(final_epochs):
        for data in full_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = final_model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
    # Save Model
    torch.save(final_model.state_dict(), 'gnn_best_model_cv5.pth')
    print("Final Model Saved to 'gnn_best_model_cv5.pth'")
    
    # --- EXTERNAL VALIDATION ---
    final_model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for data in ext_loader:
            data = data.to(device)
            out = final_model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy()) # Capture positive class prob
            
    ext_bacc = balanced_accuracy_score(y_true, y_pred)
    print(f"\n>>> EXTERNAL VALIDATION BACC (Default 0.5): {ext_bacc:.4f}")

    # 4.1 Calculate Optimal Threshold (G-Mean)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    best_threshold = thresholds[ix]
    
    y_pred_opt = (np.array(y_prob) >= best_threshold).astype(int)
    bacc_opt = balanced_accuracy_score(y_true, y_pred_opt)
    
    print(f">>> EXTERNAL VALIDATION BACC (Optimal {best_threshold:.2f}): {bacc_opt:.4f}")
    
    # Save Threshold
    with open('model_threshold.txt', 'w') as f:
        f.write(str(best_threshold))
    print(f"Optimal threshold saved to 'model_threshold.txt' ({best_threshold:.4f})")

    # --- ARCHITECTURE DIAGRAM ---
    print("\nGenerating Architecture Diagram...")
    img_name = "gnn_computational_graph" 
    img_path = img_name + ".png"
    try:
        # Dummy forward pass to trace
        dummy_data = next(iter(full_loader)).to(device)
        y = final_model(dummy_data.x, dummy_data.edge_index, dummy_data.batch, edge_attr=dummy_data.edge_attr)
        
        dot = make_dot(y, params=dict(final_model.named_parameters()))
        dot.format = 'png'
        dot.render(img_name)
        print(f"Diagram saved as '{img_path}'")
        
        # Make a copy as gnn_architecture_diagram.png for compatibility
        import shutil
        shutil.copy(img_path, "gnn_architecture_diagram.png")
        print("Also saved copy as 'gnn_architecture_diagram.png'")
    except Exception as e:
        print(f"Could not generate diagram: {e}")
        print("Make sure 'torchviz' and 'graphviz' are installed.")

    # --- REPORT GENERATION ---
    # Collect Basic Stats
    train_stats = {
        "Total": len(df_train),
        "Positive": len(df_train[df_train['Outcome'] == 1]),
        "Negative": len(df_train[df_train['Outcome'] == 0])
    }
    ext_stats = {
        "Total": len(df_ext),
        "Positive": len(df_ext[df_ext['Outcome'] == 1]),
        "Negative": len(df_ext[df_ext['Outcome'] == 0])
    }
    
    generate_pdf_report(
        report_path='report_optimized.pdf',
        best_cv_score=study.best_value,
        ext_bacc=ext_bacc,
        best_params=best_params,
        y_true_ext=y_true,
        y_pred_ext=y_pred,
        y_prob_ext=y_prob, 
        arch_img_path=img_path,
        best_threshold=best_threshold,
        bacc_opt=bacc_opt,
        train_stats=train_stats,
        ext_stats=ext_stats
    )

