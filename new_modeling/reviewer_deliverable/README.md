# Reviewer deliverable — uncertainty quantification + best model/split

Resposta ao revisor: *"...reported from a single random stratified split (N = 421). No
confidence intervals, no repeated splits, no bootstrapping. Authors must report variance
across multiple splits or at minimum bootstrap confidence intervals on all reported metrics."*

Esta pasta reúne tudo que foi pedido, para **as duas curadorias**:
- **Submetida (4.212 compostos)** — o modelo publicado (`Modeling/gnn_best_model_cv5.pth`).
- **Nova (4.201 compostos)** — modelo **retreinado** com os mesmos hiperparâmetros, sobre a
  nova curadoria (`Curation/curated_dataset.csv`), com split estratificado 90/10 mantendo a
  restrição do benchmark Odorify.

---

## Conteúdo

```
reviewer_deliverable/
├── README.md                     (este arquivo)
├── tabelas/
│   ├── RESUMO_TABELAS.md         ← comece por aqui (todas as tabelas formatadas)
│   ├── submetido_bootstrap_CIs.csv
│   ├── submetido_vs_odorify.csv
│   ├── submetido_splits_repetidos.csv
│   ├── novacuracao_bootstrap_CIs.csv
│   ├── novacuracao_splits_repetidos.csv
│   └── novacuracao_splits_por_seed.csv
├── figuras/
│   ├── submetido_forest_vs_odorify.png   (Odor-Sight vs Odorify, IC 95%)
│   ├── submetido_boxplot_15splits.png
│   └── novacuracao_boxplot_15splits.png
└── melhor_modelo/                ← melhor split por BACC (nova curadoria, seed 1009)
    ├── train_dataset.csv         (split de treino completo — 3.781; o modelo usa 85% = 3.213)
    ├── test_dataset.csv          (split de teste — 420 moléculas)
    ├── gnn_model_best.pth        (modelo treinado)
    ├── best_params.json
    ├── model_threshold.txt
    └── metrics.json
```

## Metodologia (resumo)
- **Bootstrap CIs:** 10.000 reamostras estratificadas do test set → IC 95% percentil para
  todas as métricas + ROC-AUC e PR-AUC (independentes de threshold).
- **Splits repetidos:** K=15 splits estratificados independentes, cada um retreinando o
  modelo; reporta média ± DP e faixa de 95%.
- **Threshold honesto:** escolhido no treino (CV/holdout), nunca no test set (corrige um
  vazamento do pipeline original, em que o threshold era otimizado no próprio teste).
- **Split da nova curadoria:** 90/10 estratificado, com o test set sorteado **apenas entre
  moléculas ausentes do treino do Odorify** (mesma restrição do paper). Como o arquivo
  `odorify_dataset.xlsx` não está no repositório, a exclusividade foi reconstruída a partir
  das flags `no_overlap` confiáveis do split submetido (backup em
  `new_modeling/uncertainty/_submitted_version_backup/`). As 168 moléculas genuinamente novas
  (status desconhecido no Odorify) foram, conservadoramente, mantidas no treino.

## Melhor modelo entregue (`melhor_modelo/`)
Selecionado por **maior BACC** entre os 15 splits repetidos da nova curadoria — **seed 1009**
(maior BACC registrado: 0,912). O modelo salvo aqui, reregenerado, dá **BACC 0,899; MCC 0,766;
F1 0,932; ROC-AUC 0,939** (threshold 0,716) — a pequena diferença vem do **não-determinismo do
treino em GPU**, que a média ± DP dos 15 splits já captura. Serve como **modelo final para deploy**.

> ⚠️ **Não reportar o "melhor de 15" como número principal do paper.** Isso é viés de
> seleção — justamente o problema que o revisor aponta. Para o texto, use a **média ± DP**
> (nova curadoria: BACC 0,89 ± 0,01; MCC 0,74 ± 0,02) ou os CIs de bootstrap.

## Avisos importantes
1. **Nada publicado foi sobrescrito.** O modelo e dados da versão submetida seguem intactos
   (`Modeling/gnn_best_model_cv5.pth`; backup dos CSVs em `_submitted_version_backup/`).
2. **Benchmark vs Odorify não foi regerado** para a nova curadoria — as predições do Odorify
   existem só para as 421 moléculas do teste antigo. Um teste novo exigiria rodar o Odorify
   externo nas novas moléculas.
3. **Ambiente:** o PyTorch foi trocado para o build CUDA (`2.12.0+cu126`) para usar a GPU.
4. **Scripts** (reprodutíveis): `new_modeling/newcuration_retrain/` (retreino + incerteza da nova
   curadoria) e `new_modeling/uncertainty/` (versão submetida).

## Qual curadoria usar no paper?
A exigência do revisor (CIs/splits) está atendida em **ambas**. Se o paper adotar a nova
curadoria (4.201), todos os números do manuscrito precisam ser atualizados para os desta
pasta (ver `tabelas/RESUMO_TABELAS.md`), e o benchmark Odorify precisaria ser refeito.
