# new_modeling — resposta ao revisor (quantificação de incerteza) + retreino na nova curadoria

Consolida **todos os artefatos** produzidos para responder à review:

> *"The external test set performance (BACC = 0.91, MCC = 0.79) is reported from a single
> random stratified split (N = 421). No confidence intervals, no repeated splits, no
> bootstrapping. Authors must report variance across multiple splits or at minimum bootstrap
> confidence intervals on all reported metrics."*

Cobre **duas curadorias**: a **submetida (4.212 compostos)** e a **nova (4.201)**.

## 👉 Comece por: `reviewer_deliverable/`
Pasta de entrega principal — tabelas prontas, figuras e o melhor modelo. Leia
`reviewer_deliverable/README.md` e `reviewer_deliverable/tabelas/RESUMO_TABELAS.md`.

**Mapa arquivo-a-arquivo (o que é cada arquivo):** veja [`INVENTARIO.md`](INVENTARIO.md).

## Estrutura
```
new_modeling/
├── README.md                        (este arquivo)
├── gnn_utils.py, best_params.json   (dependências dos scripts — cópias do pipeline do repo)
│
├── uncertainty/                     ► VERSÃO SUBMETIDA (4.212)
│   ├── common.py                    (modelo + métricas + bootstrap + treino)
│   ├── uncertainty_bootstrap.py     (Abordagem A: bootstrap CIs + pareado vs Odorify)
│   ├── repeated_splits.py           (Abordagem B: K=15 splits)
│   ├── REVIEWER_RESPONSE.md         (resposta detalhada)
│   ├── MANUSCRIPT_EDITS.md          (texto pronto p/ o manuscrito, com localização exata)
│   ├── results/                     (CSVs de CIs + figuras + predições por molécula)
│   └── _submitted_version_backup/   (CSVs originais da versão submetida)
│
├── newcuration_retrain/            ► NOVA CURADORIA (4.201)
│   ├── retrain_new_curation.py      (split restrito ao benchmark + retreino, HPs publicados)
│   ├── uncertainty_new_curation.py  (Abordagens A+B na nova curadoria)
│   ├── assemble_best_model.py       (regenera o melhor split por BACC)
│   ├── DATA/                        (split primário da nova curadoria — train 3.781 / test 420)
│   ├── gnn_model_newcuration.pth    (modelo primário retreinado)
│   ├── metrics.json, model_threshold.txt, report.txt
│   └── uncertainty_results/         (CIs + K=15 splits da nova curadoria)
│
├── ml_baseline/                    ► BASELINES CLÁSSICOS (RF/SVM/XGBoost)
│   ├── ml_common.py, train_ml_baselines.py  (features Morgan/ECFP4 + tuning + CIs + joblib)
│   ├── ML_BASELINE_RESULTS.md       (tabelas + interpretação + texto p/ o paper)
│   └── results/                     (comparação GNN vs clássicos + CIs + pareado + figura)
│
├── robustness/                     ► TRAIN / CV / TEST (2º revisor)
│   ├── train_cv_test_report.py      (GNN + RF/SVM/XGB: train + CV5 + test)
│   ├── ROBUSTNESS_TRAIN_CV_TEST.md  (tabela + interpretação + texto p/ o paper)
│   └── results/train_cv_test.csv
│
└── reviewer_deliverable/           ► ENTREGA CONSOLIDADA (enxuta)
    ├── README.md
    ├── tabelas/RESUMO_TABELAS.md   (tabelas inline; CSVs/figuras brutos nas pastas-fonte)
    └── melhor_modelo/  (split + .pth do melhor resultado por BACC)
```

## Resultados (média ± DP em 15 splits)
| | Submetida (4.212) | Nova curadoria (4.201) |
|---|---|---|
| BACC | 0,90 ± 0,02 | 0,886 ± 0,011 |
| MCC | 0,78 ± 0,03 | 0,745 ± 0,022 |
| F1 | 0,93 ± 0,01 | 0,926 ± 0,009 |
| ROC-AUC | 0,95 ± 0,01 | 0,949 ± 0,008 |

## Como reproduzir
Os scripts são autossuficientes se `new_modeling/` estiver em `OdorSight/` (usam `Curation/` e
`benchmark_odorify/` do repositório por caminho relativo). Ambiente: `torch 2.12.0+cu126` (GPU).
- Versão submetida: `python uncertainty/uncertainty_bootstrap.py --threshold-method cv5` e
  `python uncertainty/repeated_splits.py --seeds 15`
- Nova curadoria: `python newcuration_retrain/retrain_new_curation.py` depois
  `python newcuration_retrain/uncertainty_new_curation.py --seeds 15`
- Baselines ML: `python ml_baseline/train_ml_baselines.py` (RF/SVM/XGB em Morgan/ECFP4; salva .joblib)
- Robustez (train/CV/test): `python robustness/train_cv_test_report.py`

## Avisos
1. **Não reportar "melhor de N" como número principal** (viés de seleção). Use média ± DP / CIs.
2. **Benchmark vs Odorify não foi regerado** para a nova curadoria (exigiria rodar o Odorify
   externo nas novas moléculas de teste).
3. **Scripts da versão submetida** (`uncertainty/uncertainty_bootstrap.py`, `repeated_splits.py`)
   liam de `DATA/` e da cópia do modelo publicado, ambos removidos — para re-rodar, reaponte para
   `uncertainty/_submitted_version_backup/` e `Modeling/gnn_best_model_cv5.pth`.
