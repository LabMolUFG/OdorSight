# Baselines de ML clássico vs GNN (nova curadoria, 4.201)

Baselines RF / SVM / XGBoost treinados com **o mesmo rigor do GNN**: nova curadoria (4.201),
split restrito ao benchmark (test disjunto do Odorify, 90/10, mesmas seeds), HPs tunados por
**CV 5-fold com BACC** (Optuna, espelhando o protocolo do GNN), class-weighting, threshold
honesto (do treino), **bootstrap CIs + K=15 splits repetidos**, mesmo painel de métricas.
Duas famílias de features: **ECFP4** (Morgan, raio 2, 2048 bits) e **12 descritores
físico-químicos core** (estendem os 4 globais do GNN).

## Tabela 1 — Comparação principal (K=15 splits, média ± DP)

| Modelo | BACC | MCC | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **GNN (GAT)** | 0,886 ± 0,011 | 0,745 ± 0,022 | 0,926 ± 0,009 | 0,949 ± 0,008 | 0,978 ± 0,006 |
| RF · ECFP4 | 0,861 ± 0,011 | 0,699 ± 0,027 | 0,914 ± 0,011 | 0,934 ± 0,011 | 0,973 ± 0,006 |
| **RF · descritores** | 0,893 ± 0,012 | 0,747 ± 0,025 | 0,924 ± 0,010 | 0,948 ± 0,008 | 0,980 ± 0,004 |
| SVM · ECFP4 | 0,833 ± 0,028 | 0,621 ± 0,063 | 0,874 ± 0,038 | 0,910 ± 0,015 | 0,957 ± 0,009 |
| **SVM · descritores** | 0,888 ± 0,012 | 0,747 ± 0,024 | 0,926 ± 0,010 | 0,952 ± 0,009 | 0,982 ± 0,005 |
| XGB · ECFP4 | 0,865 ± 0,012 | 0,707 ± 0,021 | 0,916 ± 0,011 | 0,943 ± 0,008 | 0,976 ± 0,005 |
| **XGB · descritores** | 0,889 ± 0,015 | 0,740 ± 0,027 | 0,922 ± 0,010 | 0,948 ± 0,007 | 0,980 ± 0,004 |

## Tabela 2 — Comparação pareada vs GNN (mesmas moléculas de teste)
Δ = (clássico − GNN); BACC/MCC no cutoff 0,5, ROC/PR independentes de threshold.
`*` = IC 95% exclui 0 (diferença significativa).

| Modelo | Δ BACC | Δ MCC | Δ ROC-AUC | Δ PR-AUC | Significativo? |
|---|---|---|---|---|---|
| RF · descritores | −0,002 | +0,023 | +0,008 | +0,009 | não (empata) |
| RF · ECFP4 | −0,035 | −0,040 | −0,018 | −0,008 | não |
| SVM · descritores | −0,015 | +0,012 | +0,020 | +0,014 | não (empata) |
| SVM · ECFP4 | **−0,046\*** | −0,059 | −0,020 | −0,006 | **sim (pior no BACC)** |
| XGB · descritores | +0,004 | +0,026 | +0,010 | +0,009 | não (empata) |
| XGB · ECFP4 | −0,010 | +0,005 | −0,004 | −0,003 | não |

## Principais achados

1. **Os modelos clássicos com descritores empatam com o GNN.** Em RF/SVM/XGB sobre os 12
   descritores, **nenhuma** diferença vs GNN é estatisticamente significativa (todos os IC 95%
   da diferença pareada incluem 0) — em BACC, MCC, ROC-AUC e PR-AUC. XGB|desc e SVM|desc ficam
   até ligeiramente acima numericamente.
2. **Descritores > fingerprints.** As três famílias com ECFP4 são consistentemente inferiores
   às com descritores; **SVM|ECFP4 é o único modelo significativamente pior que o GNN** (BACC).
3. **Conclusão:** para esta classificação binária odorante/inodoro, a representação de grafo
   aprendida pelo GNN **não oferece vantagem preditiva estatisticamente significativa** sobre um
   modelo clássico bem tunado com descritores físico-químicos.

## Implicação para o manuscrito (importante)

Isso **não enfraquece** o paper — **fortalece**, se reposicionado com honestidade:
- O valor do Odor-Sight/GNN passa a ser a **interpretabilidade** (EdgeSHAPer, contribuições por
  ligação — que os clássicos não oferecem), o **domínio de aplicabilidade** e a **plataforma
  web integrada**, e não uma suposta superioridade de acurácia.
- Adicionar baselines clássicos **antecipa a pergunta clássica de revisor** ("compararam com
  modelos simples?") e adere às boas práticas OECD/Tropsha.

**Sugestão de subseção (Results) — "Comparison with classical ML baselines":**
> To contextualize the GNN, we trained Random Forest, SVM and XGBoost classifiers under an
> identical protocol (same curation, benchmark-restricted split, 5-fold CV hyperparameter
> tuning, class weighting, honest thresholding, and K = 15 repeated splits), using either ECFP4
> fingerprints or 12 physicochemical descriptors. On descriptor features, the classical models
> matched the GNN within statistical error (e.g., XGBoost BACC 0.889 ± 0.015 vs GNN
> 0.886 ± 0.011; all paired-bootstrap differences non-significant, 95% CIs spanning zero),
> whereas fingerprint-based models were weaker. These results indicate that, for binary
> odorant classification, the principal advantages of the graph-based model lie in
> bond-level interpretability (EdgeSHAPer) and the integrated, applicability-domain-aware web
> platform, rather than in raw predictive accuracy over well-tuned classical baselines.

## Arquivos
`results/comparison_GNN_vs_classical.csv` · `results/primary_bootstrap_CIs.csv` ·
`results/repeated_splits.csv` · `results/paired_vs_gnn.csv` · `results/best_params.json` ·
`results/comparison_BACC.png` · `results/log.txt`
