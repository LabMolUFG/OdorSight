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
| `gnn_best_model_cv5.pth` ⚙️ | **Modelo PUBLICADO** (curadoria 4.212). Cópia do original em `Modeling/`; carregado pelos scripts da versão submetida. 5,5 MB. |
| `DATA/train_dataset.csv` ⚙️📎 | Split de **treino da versão submetida** (3.791 moléculas). Lido por `uncertainty/*.py`. |
| `DATA/test_dataset.csv` ⚙️📎 | Split de **teste da versão submetida** (421). Lido por `uncertainty/*.py`. |

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
| `_submitted_version_backup/curated_dataset_auto_inspection.csv` 📊 | Curadoria **antiga** completa (4.212), preservada porque o `git pull` a removeu do repo. |

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
| `best_params.json` 📎 | Cópia dos HPs (registro de quais HPs este retreino usou). |

---

## `reviewer_deliverable/` — ENTREGA CONSOLIDADA (autossuficiente)
| Arquivo | O que é |
|---|---|
| `README.md` 📄 | Guia da entrega. |
| `tabelas/RESUMO_TABELAS.md` 📄 | **Todas as tabelas formatadas** (submetida + nova + melhor modelo). Comece aqui. |
| `tabelas/submetido_bootstrap_CIs.csv` 📎 | = `uncertainty/results/bootstrap_summary.csv`. |
| `tabelas/submetido_vs_odorify.csv` 📎 | = `uncertainty/results/paired_diff_summary.csv`. |
| `tabelas/submetido_splits_repetidos.csv` 📎 | = `uncertainty/results/repeated_splits_summary.csv`. |
| `tabelas/novacuracao_bootstrap_CIs.csv` 📎 | = `newcuration_retrain/uncertainty_results/bootstrap_summary.csv`. |
| `tabelas/novacuracao_splits_repetidos.csv` 📎 | = `newcuration_retrain/uncertainty_results/repeated_splits_summary.csv`. |
| `tabelas/novacuracao_splits_por_seed.csv` 📎 | = `newcuration_retrain/uncertainty_results/repeated_splits_per_seed.csv`. |
| `figuras/submetido_forest_vs_odorify.png` 📎 | = `uncertainty/results/bootstrap_forest.png`. |
| `figuras/submetido_boxplot_15splits.png` 📎 | = `uncertainty/results/repeated_splits_distribution.png`. |
| `figuras/novacuracao_boxplot_15splits.png` 📎 | = `newcuration_retrain/uncertainty_results/repeated_splits_distribution.png`. |
| `melhor_modelo/train_dataset.csv` 📊 | Split de treino do **melhor modelo** (seed 1009, 3.781; treino usa 85% = 3.213). **Não é duplicata** dos outros splits (seed diferente). |
| `melhor_modelo/test_dataset.csv` 📊 | Split de teste do melhor modelo (420, seed 1009). |
| `melhor_modelo/gnn_model_best.pth` 📊 | **Melhor modelo** por BACC (seed 1009). Para deploy. 5,5 MB. |
| `melhor_modelo/metrics.json` 📊 | Métricas do melhor modelo (BACC 0,899; threshold honesto 0,716). |
| `melhor_modelo/model_threshold.txt` 📊 | Threshold operacional (0,716). |
| `melhor_modelo/best_params.json` 📎 | Cópia dos HPs. |

---

## Redundância (proposital — mantida a seu pedido)
São **12 grupos** de arquivos idênticos, todos pequenos, por dois motivos legítimos:
1. **Entrega autossuficiente** — as 6 tabelas e 3 figuras em `reviewer_deliverable/` são cópias
   renomeadas dos CSVs/PNGs de `*/results/`, para quem receber só a pasta de entrega ter tudo.
2. **Pastas autodocumentadas** — `best_params.json` (×3) e o split submetido (`DATA/` ≡
   `_submitted_version_backup/`) aparecem onde cada script/análise precisa deles.

**Os 16,5 MB de `.pth` NÃO são redundância:** são **3 modelos diferentes** — publicado (4.212),
primário da nova curadoria (seed 42) e melhor da nova curadoria (seed 1009). O único que duplica
algo externo é `gnn_best_model_cv5.pth` (cópia do publicado que também está em `Modeling/`).
