# Baselines de ML clássico (Morgan/ECFP4) vs GNN — nova curadoria (4.201)

Baselines RF / SVM / XGBoost treinados com **o mesmo rigor do GNN**: nova curadoria (4.201),
split restrito ao benchmark (test disjunto do Odorify, 90/10, mesmas seeds), HPs tunados por
**CV 5-fold com BACC** (Optuna), class-weighting, threshold honesto (do treino),
**bootstrap CIs + K=15 splits repetidos**, mesmo painel de métricas.
Features: **Morgan/ECFP4** (fingerprints, raio 2, 2048 bits).

## Tabela 1 — Comparação principal (K=15 splits, média ± DP)

| Modelo | BACC | MCC | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **GNN (GAT)** | **0,886 ± 0,011** | **0,745 ± 0,022** | **0,926 ± 0,009** | **0,949 ± 0,008** | **0,978 ± 0,006** |
| RF · ECFP4 | 0,861 ± 0,011 | 0,699 ± 0,027 | 0,914 ± 0,011 | 0,934 ± 0,011 | 0,973 ± 0,006 |
| SVM · ECFP4 | 0,833 ± 0,028 | 0,621 ± 0,063 | 0,874 ± 0,038 | 0,910 ± 0,015 | 0,957 ± 0,009 |
| XGB · ECFP4 | 0,865 ± 0,012 | 0,707 ± 0,021 | 0,916 ± 0,011 | 0,943 ± 0,008 | 0,976 ± 0,005 |

## Tabela 2 — Comparação pareada vs GNN (mesmas moléculas de teste)
Δ = (clássico − GNN); BACC/MCC no cutoff 0,5, ROC/PR independentes de threshold.
`*` = IC 95% exclui 0 (diferença significativa). Δ < 0 ⇒ GNN melhor.

| Modelo | Δ BACC | Δ MCC | Δ ROC-AUC | Δ PR-AUC | Significativo? |
|---|---|---|---|---|---|
| RF · ECFP4 | −0,035 | −0,040 | −0,018 | −0,008 | não (GNN ~ melhor) |
| SVM · ECFP4 | **−0,046\*** | −0,059 | −0,020 | −0,006 | **sim — GNN melhor** |
| XGB · ECFP4 | −0,010 | +0,005 | −0,004 | −0,003 | não (praticamente empate) |

## Principais achados

1. **O GNN fica no topo em todas as métricas.** Contra os três baselines de fingerprint, o GNN
   tem a maior BACC/MCC/F1/ROC-AUC/PR-AUC (K=15).
2. **Significância:** o GNN é **estatisticamente superior ao SVM-ECFP** (ΔBACC −0,046; IC 95%
   0,00–0,09 a favor do GNN). Contra **RF-ECFP** (Δ −0,035) e **XGB-ECFP** (Δ −0,010) o GNN é
   melhor **numericamente**, mas sem significância (ICs incluem 0) — **XGB-ECFP é o mais
   próximo**, praticamente empatado.
3. **Conclusão:** sobre fingerprints Morgan, a representação de grafo aprendida pelo GNN
   **iguala ou supera** os baselines clássicos, sendo **significativamente superior ao SVM** e
   ≥ RF/XGBoost. Somando à **interpretabilidade em nível de ligação (EdgeSHAPer)** e à plataforma
   integrada, isso sustenta a escolha do modelo de grafo.

## Sugestão de subseção (Results) — "Comparison with classical ML baselines"
> To contextualize the GNN, we trained Random Forest, SVM and XGBoost classifiers on
> Morgan/ECFP4 fingerprints under an identical protocol (same curation, benchmark-restricted
> split, 5-fold CV hyperparameter tuning, class weighting, honest thresholding, and K = 15
> repeated splits). Across splits, the GNN achieved the highest scores on every metric
> (BACC 0.886 ± 0.011 vs RF 0.861, SVM 0.833, XGBoost 0.865), significantly outperforming the
> SVM (paired-bootstrap ΔBACC = 0.046, 95% CI 0.00–0.09), with smaller, non-significant margins
> over RF and XGBoost. Together with its bond-level interpretability (EdgeSHAPer) and the
> integrated, applicability-domain-aware platform, these results support the graph-based model.

## Modelos serializados (`results/models/`)
`joblib`, um por algoritmo para o **split primário** (seed 42) e para o **melhor split** (maior
BACC no K=15). Metadados (threshold + métricas) em `results/models/models_meta.json`.

| Arquivo | Split | Threshold | BACC (test) |
|---|---|---|---|
| `rf_ecfp4_primary.joblib` | primário | 0,650 | 0,861 |
| `rf_ecfp4_bestsplit_seed1009.joblib` | melhor (seed 1009) | 0,711 | 0,878 |
| `svm_ecfp4_primary.joblib` | primário | 0,670 | 0,857 |
| `svm_ecfp4_bestsplit_seed1008.joblib` | melhor (seed 1008) | 0,663 | 0,871 |
| `xgb_ecfp4_primary.joblib` | primário | 0,704 | 0,891 |
| `xgb_ecfp4_bestsplit_seed1001.joblib` | melhor (seed 1001) | 0,784 | 0,885 |

## Arquivos
`results/comparison_GNN_vs_classical.csv` · `results/primary_bootstrap_CIs.csv` ·
`results/repeated_splits.csv` · `results/paired_vs_gnn.csv` · `results/best_params.json` ·
`results/comparison_BACC.png` · `results/models/` · `results/log.txt`.
Scripts: `train_ml_baselines.py`, `ml_common.py`.
