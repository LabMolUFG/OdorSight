# Methodology: Optimized Graph Neural Network for Olfactory Prediction

## 1. Objective

The primary objective of this study was to develop a high-performance Graph Neural Network (GNN) capable of predicting olfactory properties from molecular structures. The target performance metric was a **Balanced Accuracy (BACC)** on an external, unseen validation set.

## 2. Dataset Preparation

Data was sourced from two primary files:

1.  **Training Set (`curated_dataset_final.csv`)**: A curated dataset used for model training and hyperparameter optimization.
2.  **External Validation Set (`moleculas_unicas.csv`)**: A distinct dataset of unique molecules served as a hold-out test set to verify generalizability.

### Preprocessing Pipeline

- **Canonicalization**: All SMILES strings were standardized using RDKit to ensure consistent molecular representation.
- **Leakage Prevention**: A strict overlap check was performed. Any molecule in the Training Set that also appeared in the External Validation Set was removed from the Training Set to strictly enforce the independence of the validation phase.
- **Graph Construction**: Molecules were converted into graph structures using `PyTorch Geometric`.
  - **Nodes**: Atoms (featurized by atomic number, chirality, etc.).
  - **Edges**: Chemical bonds (featurized by bond type, conjugation, etc.).

## 3. Model Architecture

We implemented a flexible **Graph Attention Network (GAT)** architecture (`GNN_Optimized`), chosen for its ability to weigh the importance of neighboring atoms dynamically.

- **Core Layers**: `GATv2Conv` (Graph Attention v2) layers.
- **Global Pooling**: Global Mean Pooling to aggregate node features into a graph-level vector.
- **Classifier**: A 2-layer Multi-Layer Perceptron (MLP) with ReLU activation and Dropout for regularization.
- **Dynamic Hyperparameters**: The architecture allows flexible configuration of:
  - Number of GAT Layers (3 to 6).
  - Hidden Channel Dimensions (32, 64, 128, 256).
  - Attention Heads (2, 4, 8).
  - Dropout Rates (0.1 to 0.5).

## 4. Optimization Strategy (Phase 1)

To maximize model performance, we employed **Bayesian Optimization** using the `Optuna` framework.

### Stratified 5-Fold Cross-Validation

Instead of a simple train-test split, we utilized **Stratified 5-Fold Cross-Validation** on the `curated_dataset_final.csv`.

- **Stratification**: Ensures the ratio of positive/negative classes is preserved in every fold, which is critical for imbalanced datasets.
- **Process**:
  1.  The optimization engine suggests a set of hyperparameters (Learning Rate, Weight Decay, Batch Size, Model Depth, etc.).
  2.  The model is trained ~5 times (once for each fold).
  3.  **Metric**: The objective function maximizes the **Average Balanced Accuracy** across the 5 validation folds.

This robust approach prevents overfitting to a specific subset of the training data and selects hyperparameters that generalize well.

## 5. Final Model & External Validation (Phase 2)

The external validation set (`moleculas_unicas.csv`) was **strictly isolated** during the optimization phase (Phase 1). It was never used to select hyperparameters.

The validation procedure followed these specific steps:

1.  **Selection of Best Hyperparameters**: The parameters yielding the highest _Average CV Score_ in Phase 1 were selected.
2.  **Full Retraining**: A completely new model instance was initialized with these optimal parameters. This model was then trained on the **entirety** of the `curated_dataset_final.csv` dataset (combining all 5 folds, effectively using 100% of the available training data).
3.  **Threshold Optimization**: We utilized **G-Mean Optimization** on this final model's predictions to determine the optimal decision threshold.
4.  **Final External Assessment**: Only _after_ retraining was complete, the model was evaluated on the `moleculas_unicas.csv` dataset. This ensures that the reported metrics (BACC) reflect the model's true generalization capability on purely unseen data.

## 6. Software & Libraries

- **Deep Learning**: PyTorch, PyTorch Geometric.
- **Chemoinformatics**: RDKit.
- **Optimization**: Optuna.
- **Analysis & Visualization**: Scikit-Learn, Matplotlib, Torchviz.
