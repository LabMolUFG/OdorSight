# INVENTÁRIO — o que é cada arquivo em `new_modeling/`

Mapa completo, arquivo a arquivo. Legenda:
**⚙️** = lido/necessário por algum script (não apagar) · **📎** = cópia idêntica de outro
arquivo (proposital — ver "Redundância" no fim) · **📊** = resultado (dado) · **📄** = documento.

Para uma leitura rápida do que importa: `README.md` (raiz) → `reviewer_deliverable/`.

---

## Raiz `new_modeling/`
| Arquivo | O que é |
|---|---|
| `README.md` 📄 | Visão geral da pasta (as duas curadorias, estrutura, como reproduzir). |
| `INVENTARIO.md` 📄 | Este arquivo. |
| `gnn_utils.py` ⚙️ | Featurização molécula→grafo (átomos/ligações/descritores). Cópia do pipeline do repo; usado por `common.py`. |
| `best_params.json` ⚙️📎 | Hiperparâmetros publicados (GAT, 6 camadas, 256, etc.). Lido por `common.load_params()`. |

---

## `uncertainty/` — VERSÃO SUBMETIDA (curadoria 4.212)
| Arquivo | O que é |
|---|---|
| `common.py` ⚙️ | Núcleo compartilhado: classe do modelo GNN (cópia fiel), métricas, bootstrap, treino, seleção de threshold. Importado por todos os outros scripts. |
| `uncertainty_bootstrap.py` ⚙️ | **Abordagem A**: recarrega o modelo publicado, gera CIs de bootstrap (10.000) e a comparação **pareada vs Odorify**. |
| `repeated_splits.py` ⚙️ | **Abordagem B**: K=15 splits estratificados, retreinando o modelo em cada. |
| `REVIEWER_RESPONSE.md` 📄 | Resposta detalhada ao revisor (reprodução, CIs, splits, correção do threshold). |
| `MANUSCRIPT_EDITS.md` 📄 | **Texto pronto** para o manuscrito, com âncoras exatas de onde inserir (versão submetida). |
| `results/bootstrap_summary.csv` 📊 | Ponto + IC 95% de cada métrica em 3 thresholds (0,5 / ótimo-teste / honesto 0,734). |
| `results/paired_diff_summary.csv` 📊 | Diferença Odor-Sight − Odorify (IC + p bootstrap). |
| `results/repeated_splits_summary.csv` 📊 | Média ± DP das métricas nos 15 splits. |
| `results/repeated_splits_per_seed.csv` 📊 | Métricas por seed (15 linhas). |
| `results/test_predictions.csv` 📊 | Predições por molécula (y_true, prob Odor-Sight, prob/pred Odorify) — base do bootstrap. |
| `results/bootstrap_forest.png` 📊 | Forest plot Odor-Sight vs Odorify (IC 95%). |
| `results/repeated_splits_distribution.png` 📊 | Boxplot das métricas nos 15 splits. |
| `results/*_log.txt`, `protocol_log.txt` 📄 | Logs das execuções. *Obs.: contêm o caminho absoluto do run original (`...Modeling/uncertainty/results`); é só texto de log, não afeta reprodução.* |
| `_submitted_version_backup/train_dataset.csv` ⚙️ | Backup do split submetido (idem `DATA/`). Lido pelos scripts da **nova** curadoria p/ recuperar as flags `no_overlap` (quem está no Odorify). |
| `_submitted_version_backup/test_dataset.csv` ⚙️ | idem (teste submetido). |

---

## `newcuration_retrain/` — NOVA CURADORIA (4.201)
| Arquivo | O que é |
|---|---|
| `retrain_new_curation.py` ⚙️ | Constrói o split restrito ao benchmark sobre a nova curadoria e **retreina** com os HPs publicados. |
| `uncertainty_new_curation.py` ⚙️ | Abordagens A+B na nova curadoria (bootstrap CIs + K=15 splits). |
| `assemble_best_model.py` ⚙️ | Regenera o **melhor split por BACC** (seed 1009) e o salva na entrega. |
| `DATA/train_dataset.csv` 📊 | Split de **treino primário** da nova curadoria (3.781, seed 42). |
| `DATA/test_dataset.csv` 📊 | Split de **teste primário** da nova curadoria (420, seed 42). |
| `gnn_model_newcuration.pth` 📊 | Modelo **primário** da nova curadoria (seed 42). 5,5 MB. |
| `report.txt` / `metrics.json` / `model_threshold.txt` 📄📊 | Relatório do retreino-referência. ⚠️ **Reporta métricas em 0,5 e no threshold ótimo-no-teste (0,764)** — isto espelha o pipeline ORIGINAL só para comparação direta; **não** é o número honesto. Os números honestos (threshold do treino) estão em `uncertainty_results/` e em `melhor_modelo/`. |
| `uncertainty_results/bootstrap_summary.csv` 📊 | CIs de bootstrap da nova curadoria (threshold honesto 0,94). |
| `uncertainty_results/repeated_splits_summary.csv` 📊 | Média ± DP nos 15 splits (nova curadoria). |
| `uncertainty_results/repeated_splits_per_seed.csv` 📊 | Métricas por seed (15 linhas; seed 1009 = maior BACC). |
| `uncertainty_results/repeated_splits_distribution.png` 📊 | Boxplot (nova curadoria). |
| `uncertainty_results/log.txt` 📄 | Log da execução (mesma obs. de caminho absoluto). |

---

## `ml_baseline/` — BASELINES CLÁSSICOS (RF/SVM/XGBoost, Morgan/ECFP4)
| Arquivo | O que é |
|---|---|
| `ml_common.py` ⚙️ | Featurização Morgan/ECFP4 (2048 bits), construção do pool/split restrito, fábricas de modelo (RF/SVM/XGB), tuning Optuna (CV5-BACC) e threshold honesto. Importa `common.py` da `uncertainty/`. |
| `train_ml_baselines.py` ⚙️ | Runner: por algoritmo, tuna HPs, treina no split primário, gera bootstrap CIs + K=15 splits + **pareado vs GNN**, e salva o `.joblib`. |
| `ML_BASELINE_RESULTS.md` 📄 | Tabelas + interpretação (GNN vs clássicos) + **texto sugerido** para o manuscrito. |
| `results/comparison_GNN_vs_classical.csv` 📊 | Tabela mestre (K=15, média ± DP): GNN vs RF/SVM/XGB. |
| `results/primary_bootstrap_CIs.csv` 📊 | Ponto + IC 95% de cada métrica no test primário (por algoritmo). |
| `results/repeated_splits.csv` 📊 | Média ± DP nos 15 splits (por algoritmo). |
| `results/paired_vs_gnn.csv` 📊 | Diferença (clássico − GNN) pareada no mesmo teste (IC + p). |
| `results/comparison_BACC.png` 📊 | Barras de BACC: GNN vs clássicos (± DP). |
| `results/best_params.json` 📊 | HPs tunados de cada algoritmo (RF/SVM/XGB) + CV5-BACC. |
| `results/log.txt` 📄 | Log da execução. |
| `results/models/rf_ecfp4_primary.joblib` 📊 | Modelo **RF** treinado (split primário, Morgan/ECFP4). Deployável. |
| `results/models/svm_ecfp4_primary.joblib` 📊 | Modelo **SVM** treinado (split primário). Deployável. |
| `results/models/xgb_ecfp4_primary.joblib` 📊 | Modelo **XGBoost** treinado (split primário). Deployável. |
| `results/models/models_meta.json` 📊 | Metadados dos 3 modelos (threshold + métricas de teste). |

---

## `robustness/` — TRAIN / CV / TEST (2º revisor)
| Arquivo | O que é |
|---|---|
| `train_cv_test_report.py` ⚙️ | Gera, no split primário, métricas de **treino**, **CV 5-fold** (média ± DP) e **teste** para o GNN e os 3 baselines (RF/SVM/XGB). |
| `ROBUSTNESS_TRAIN_CV_TEST.md` 📄 | Tabela train/CV/test + interpretação (robustez, sem overfitting) + texto p/ o manuscrito. |
| `results/train_cv_test.csv` 📊 | Tabela completa (8 métricas × modelo × partição). |
| `results/log.txt` 📄 | Log da execução. |

---

## `reviewer_deliverable/` — ENTREGA CONSOLIDADA (autossuficiente)
| Arquivo | O que é |
|---|---|
| `README.md` 📄 | Guia da entrega. |
| `tabelas/RESUMO_TABELAS.md` 📄 | **Todas as tabelas formatadas** (submetida + nova + baselines ML + melhor modelo). Comece aqui. CSVs/figuras brutos ficam nas pastas-fonte. |
| `melhor_modelo/train_dataset.csv` 📊 | Split de treino do **melhor modelo** (seed 1009, 3.781; treino usa 85% = 3.213). **Não é duplicata** dos outros splits (seed diferente). |
| `melhor_modelo/test_dataset.csv` 📊 | Split de teste do melhor modelo (420, seed 1009). |
| `melhor_modelo/gnn_model_best.pth` 📊 | **Melhor modelo** por BACC (seed 1009). Para deploy. 5,5 MB. |
| `melhor_modelo/metrics.json` 📊 | Métricas do melhor modelo (BACC 0,899; threshold honesto 0,716). |
| `melhor_modelo/model_threshold.txt` 📊 | Threshold operacional (0,716). |
| `melhor_modelo/best_params.json` 📎 | Cópia dos HPs. |

---

## Redundância (após limpeza)
As duplicatas foram removidas: as tabelas/figuras que copiavam `*/results/` saíram do
`reviewer_deliverable/` (agora ele aponta para as pastas-fonte), a cópia do modelo publicado e o
`curated_dataset_auto_inspection.csv` foram apagados. Resta apenas `best_params.json` em
`melhor_modelo/` (para a pasta de deploy ser autossuficiente) e o split submetido em
`_submitted_version_backup/` (lido pelos scripts da nova curadoria).

**Os `.pth` NÃO são redundância:** são **2 modelos diferentes** — primário da nova curadoria
(seed 42, `newcuration_retrain/`) e melhor da nova curadoria (seed 1009, `melhor_modelo/`). O
modelo publicado (4.212) permanece só em `Modeling/gnn_best_model_cv5.pth`.
