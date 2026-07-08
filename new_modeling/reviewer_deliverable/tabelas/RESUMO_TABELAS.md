# Resumo das tabelas — quantificação de incerteza (resposta ao revisor)

Bootstrap = 10.000 reamostras estratificadas do test set; splits repetidos = K=15.
Threshold "honesto" = derivado do treino (CV/holdout), nunca do test set.

---

## A. Versão SUBMETIDA (curadoria 4.212) — modelo publicado

**Tabela A1 — Test set (N=421): estimativa (IC 95% bootstrap) e distribuição em 15 splits**

| Métrica | Estimativa (IC 95%) | 15 splits (média ± DP) |
|---|---|---|
| BACC | 0,90 (0,87–0,94) | 0,90 ± 0,02 |
| MCC | 0,79 (0,73–0,85) | 0,78 ± 0,03 |
| Precision | 0,95 (0,93–0,97) | 0,96 ± 0,02 |
| Recall | 0,92 (0,89–0,95) | 0,90 ± 0,02 |
| Specificity | 0,89 (0,83–0,94) | 0,91 ± 0,04 |
| F1 | 0,94 (0,92–0,96) | 0,93 ± 0,01 |
| ROC-AUC | 0,95 (0,93–0,97) | 0,95 ± 0,01 |
| PR-AUC | 0,98 (0,97–0,99) | 0,98 ± 0,01 |

**Tabela A2 — Odor-Sight vs Odorify (mesmo test set, bootstrap pareado)**

| Métrica | Δ (Odor-Sight − Odorify) | IC 95% | p |
|---|---|---|---|
| BACC | +0,18 | (0,13–0,22) | <0,001 |
| MCC | +0,22 | (0,14–0,31) | <0,001 |
| Precision | +0,13 | (0,10–0,16) | <0,001 |
| Recall | −0,05 | (−0,08, −0,03) | <0,001 |
| Specificity | +0,41 | (0,32–0,50) | <0,001 |
| F1 | +0,05 | (0,02–0,07) | <0,001 |
| ROC-AUC | +0,13 | (0,09–0,17) | <0,001 |
| PR-AUC | +0,09 | (0,07–0,12) | <0,001 |

---

## B. NOVA curadoria (4.201) — modelo retreinado (mesmos HPs, split restrito ao benchmark)

**Tabela B1 — Test set (N=420): estimativa (IC 95% bootstrap) e distribuição em 15 splits**

| Métrica | Estimativa (IC 95%) | 15 splits (média ± DP) |
|---|---|---|
| BACC | 0,89 (0,86–0,92) | 0,886 ± 0,011 |
| MCC | 0,72 (0,65–0,78) | 0,745 ± 0,022 |
| Precision | 0,97 (0,95–0,99) | 0,950 ± 0,011 |
| Recall | 0,85 (0,81–0,89) | 0,904 ± 0,021 |
| Specificity | 0,93 (0,88–0,97) | 0,869 ± 0,032 |
| F1 | 0,91 (0,88–0,93) | 0,926 ± 0,009 |
| ROC-AUC | 0,94 (0,91–0,97) | 0,949 ± 0,008 |
| PR-AUC | 0,97 (0,95–0,99) | 0,978 ± 0,006 |

> Nota: o threshold honesto da nova curadoria saiu alto (0,94), puxando para precisão
> (recall menor) na coluna de bootstrap. A coluna de 15 splits (threshold por split) é o
> resumo mais representativo do desempenho. **Benchmark pareado vs Odorify não foi regerado**
> para a nova curadoria (exigiria rodar o Odorify externo nas novas moléculas de teste).

**Tabela B2 — Melhor split por BACC (seed 1009) → modelo entregue em `melhor_modelo/`**
(threshold operacional 0,716; métricas do modelo efetivamente salvo)

| Métrica | Valor |
|---|---|
| BACC | 0,899 |
| MCC | 0,766 |
| Precision | 0,959 |
| Recall | 0,906 |
| Specificity | 0,893 |
| F1 | 0,932 |
| ROC-AUC | 0,939 |
| PR-AUC | 0,969 |

> **Nota (não-determinismo de GPU):** o seed 1009 teve o **maior BACC registrado** entre os
> 15 splits (0,912). Ao reregenerar o modelo, o treino em CUDA não é bit-determinístico, então
> o modelo salvo dá BACC 0,899. Essa diferença (0,912 → 0,899) É o próprio ruído de treino que
> a média ± DP já captura — mais um motivo para reportar a distribuição (Tabela B1), não um
> único valor.
>
> ⚠️ Além disso, "melhor de N" é o modelo final para deploy, **não** o número de manchete do
> paper (viés de seleção — exatamente o que o revisor critica). Para o texto, use a Tabela B1.

---

## C. Baselines de ML clássico vs GNN (nova curadoria)

RF/SVM/XGBoost com o **mesmo rigor** (split restrito, tuning CV5-BACC, K=15, bootstrap CIs,
threshold honesto). Features: ECFP4 (2048 bits) e 12 descritores físico-químicos.

| Modelo | BACC (K=15) | MCC | ROC-AUC | vs GNN (pareado) |
|---|---|---|---|---|
| **GNN (GAT)** | 0,886 ± 0,011 | 0,745 ± 0,022 | 0,949 ± 0,008 | — |
| RF · descritores | 0,893 ± 0,012 | 0,747 ± 0,025 | 0,948 ± 0,008 | empata (n.s.) |
| SVM · descritores | 0,888 ± 0,012 | 0,747 ± 0,024 | 0,952 ± 0,009 | empata (n.s.) |
| XGB · descritores | 0,889 ± 0,015 | 0,740 ± 0,027 | 0,948 ± 0,007 | empata (n.s.) |
| RF · ECFP4 | 0,861 ± 0,011 | 0,699 ± 0,027 | 0,934 ± 0,011 | n.s. |
| SVM · ECFP4 | 0,833 ± 0,028 | 0,621 ± 0,063 | 0,910 ± 0,015 | **pior (BACC\*)** |
| XGB · ECFP4 | 0,865 ± 0,012 | 0,707 ± 0,021 | 0,943 ± 0,008 | n.s. |

> **Achado:** modelos clássicos com descritores **empatam** com o GNN — nenhuma diferença
> pareada é significativa (ICs 95% incluem 0). Fingerprints < descritores. O valor do GNN está
> na **interpretabilidade** (EdgeSHAPer) e na plataforma, não em acurácia superior. Detalhes +
> texto sugerido para o paper: `../../ml_baseline/ML_BASELINE_RESULTS.md`.

---

### Arquivos-fonte (CSV completos nesta pasta)
- `submetido_bootstrap_CIs.csv`, `submetido_vs_odorify.csv`, `submetido_splits_repetidos.csv`
- `novacuracao_bootstrap_CIs.csv`, `novacuracao_splits_repetidos.csv`, `novacuracao_splits_por_seed.csv`
- `gnn_vs_ml_baselines.csv`, `ml_baselines_paired_vs_gnn.csv`
