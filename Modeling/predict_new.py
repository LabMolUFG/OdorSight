import torch
import json
import argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F
# Note we import from optimize_gnn_cv5 since this is the colab2 folder context
from optimize_gnn_cv5 import GNN_Optimized, canonicalize_smiles, smiles_to_data
from torch_geometric.loader import DataLoader
from rdkit import Chem

# Suppress RDKit Warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def load_prediction_artifacts(model_path='gnn_best_model_cv5.pth', params_path='best_params.json', threshold_path='model_threshold.txt'):
    # 1. Load Params
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        print(f"Loaded params from {params_path}")
    except FileNotFoundError:
        print(f"Error: {params_path} not found. Run optimization first.")
        exit(1)

    # 2. Load Threshold
    try:
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
        print(f"Loaded Optimal Threshold: {threshold:.4f}")
    except FileNotFoundError:
        print("Warning: Threshold file not found. Using default 0.5.")
        threshold = 0.5

    # 3. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Infer input dim
    dummy = smiles_to_data("C")
    num_node_features = dummy.num_node_features
    edge_dim = dummy.edge_attr.shape[1] if (dummy.edge_attr is not None and len(dummy.edge_attr.shape) > 1) else 0

    model = GNN_Optimized(
        num_node_features=num_node_features,
        hidden_channels=params['hidden_channels'],
        num_classes=2,
        heads=params['heads'],
        edge_dim=edge_dim,
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        model_type=params['model_type']
    ).to(device)
    
    if not torch.cuda.is_available():
         model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))
        
    model.eval()
    
    return model, threshold, device

def predict_smiles(model, threshold, device, smiles_list):
    results = []
    
    for smiles in smiles_list:
        canon = canonicalize_smiles(smiles)
        if not canon:
            results.append({"SMILES": smiles, "Valid": False, "Prediction": "Error", "Prob": 0.0})
            continue
            
        data = smiles_to_data(canon)
        if not data:
            results.append({"SMILES": smiles, "Valid": False, "Prediction": "Error", "Prob": 0.0})
            continue
            
        data = data.to(device)
        
        # Batch of 1
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
        
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
            probs = F.softmax(out, dim=1) # [0.1, 0.9]
            prob_odor = probs[0, 1].item() # Prob of class 1
            
        # APPLY OPTIMAL THRESHOLD
        is_odor = prob_odor >= threshold
        label = "ODOR" if is_odor else "NO ODOR"
        
        results.append({
            "SMILES": smiles,
            "Valid": True, 
            "Prediction": label,
            "Prob": prob_odor,
            "Threshold": threshold
        })
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict utilizing the Optimized GNN Model")
    parser.add_argument("smiles", nargs='?', help="SMILES string to predict")
    parser.add_argument("--file", help="CSV file containing SMILES column")
    args = parser.parse_args()

    # Note: Model filename is 'gnn_best_model_cv5.pth' in colab2
    model, threshold, device = load_prediction_artifacts(model_path='gnn_best_model_cv5.pth')

    input_smiles = []
    if args.smiles:
        input_smiles.append(args.smiles)
    elif args.file:
        df = pd.read_csv(args.file)
        if 'SMILES' in df.columns:
            input_smiles = df['SMILES'].tolist()
        else:
            print("Error: CSV must have a 'SMILES' column.")
            exit(1)
    else:
        # Example if nothing provided
        print("\nNo input provided. Running test example (Vanillin)...")
        input_smiles = ["Oc1c(OC)cc(C=O)cc1"] 

    print(f"\nRunning predictions on {len(input_smiles)} molecules...\n")
    preds = predict_smiles(model, threshold, device, input_smiles)
    
    # Print Table
    print(f"{'SMILES':<40} | {'PROB':<8} | {'THR':<6} | {'PREDICTION'}")
    print("-" * 75)
    for res in preds:
        if res['Valid']:
            print(f"{res['SMILES']:<40} | {res['Prob']:.4f}   | {res['Threshold']:.2f}   | {res['Prediction']}")
        else:
            print(f"{res['SMILES']:<40} | INVALID SMILES")
