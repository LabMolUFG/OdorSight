# Robustez — estatísticas de Treino / CV / Teste (nova curadoria, 4.201)

Atende ao comentário do revisor: *"...the authors have evaluated their models only based on the
test set performance. It is also desirable to demonstrate the model's robustness by reporting
both training and CV statistics."*

Para cada modelo, no **mesmo split primário** (seed 42) da nova curadoria:
- **Train** — métricas do modelo ajustado no próprio treino (in-sample);
- **CV** — validação cruzada estratificada **5-fold** no treino, **média ± DP entre os folds**;
- **Test** — conjunto de teste externo (disjunto do Odorify).

Métricas dependentes de limiar no **cutoff fixo 0,5** (comparação limpa entre as três partições);
ROC-AUC e PR-AUC são independentes de limiar. GNN com os HPs publicados; RF/SVM/XGB com os HPs
tunados (`ml_baseline/results/best_params.json`). CV do GNN = 5 modelos treinados por fold.

## Tabela — Train / CV / Test

| Modelo | Partição | BACC | MCC | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **GNN (GAT)** | Train | 0,987 | 0,929 | 0,988 | 0,999 | 1,000 |
| | **CV (5-fold)** | **0,931 ± 0,016** | **0,819 ± 0,03** | 0,971 ± 0,00 | 0,977 ± 0,01 | 0,994 ± 0,00 |
| | Test | 0,889 | 0,747 | 0,927 | 0,941 | 0,971 |
| **RF (ECFP4)** | Train | 0,962 | 0,879 | 0,980 | 0,996 | 0,999 |
| | **CV (5-fold)** | **0,889 ± 0,01** | **0,754 ± 0,03** | 0,962 ± 0,00 | 0,958 ± 0,01 | 0,991 ± 0,00 |
| | Test | 0,854 | 0,708 | 0,922 | 0,922 | 0,963 |
| **SVM (ECFP4)** | Train | 0,974 | 0,950 | 0,993 | 0,992 | 0,998 |
| | **CV (5-fold)** | **0,844 ± 0,02** | **0,675 ± 0,03** | 0,950 ± 0,00 | 0,932 ± 0,01 | 0,984 ± 0,00 |
| | Test | 0,843 | 0,689 | 0,917 | 0,921 | 0,966 |
| **XGB (ECFP4)** | Train | 0,970 | 0,885 | 0,981 | 0,994 | 0,999 |
| | **CV (5-fold)** | **0,897 ± 0,01** | **0,758 ± 0,03** | 0,961 ± 0,00 | 0,965 ± 0,01 | 0,993 ± 0,00 |
| | Test | 0,879 | 0,753 | 0,933 | 0,937 | 0,968 |

## Interpretação (robustez)
1. **Padrão esperado train > CV > test** em todos os modelos — sem inversões nem anomalias.
2. **A CV acompanha o teste** (sem colapso de generalização): BACC de CV 0,84–0,93 e de teste
   0,84–0,89; ROC-AUC de CV 0,93–0,98 e de teste 0,92–0,94. As estatísticas de CV são, portanto,
   estimativas **representativas** — os números de teste **não** são fruto de overfitting.
3. **GNN:** menor gap train→CV (0,987 → 0,931) e a **maior** CV/teste entre todos os modelos.
4. **SVM** é o que mais overfitta o treino (train BACC 0,974 vs CV 0,844), embora CV ≈ teste
   (0,844 vs 0,843) — a CV capturou bem o desempenho real.
5. **Nota de distribuição:** a CV é feita no *treino* (que inclui moléculas sobrepostas ao
   Odorify); o *teste* é o hold-out restrito (só moléculas disjuntas do Odorify). Por isso a CV
   fica ligeiramente acima do teste — comportamento honesto e esperado, não overfitting.

## Sugestão de texto (Methods/Results)
> To assess robustness and rule out overfitting, training, 5-fold cross-validation, and
> external-test statistics were computed for all models (fixed 0.5 threshold). For the GNN,
> balanced accuracy was 0.99 (train), 0.93 ± 0.02 (CV), and 0.89 (test), with ROC-AUC of
> 1.00 / 0.98 ± 0.01 / 0.94, respectively; the classical baselines followed the same pattern
> (e.g., XGBoost CV BACC 0.90 ± 0.01, test 0.88). Cross-validation statistics closely tracked
> external-test performance across all models, indicating that the reported test metrics are
> representative and not an artefact of overfitting.

## Arquivos
`results/train_cv_test.csv` (tabela completa, 8 métricas) · `results/log.txt` · script:
`train_cv_test_report.py` (reproduzível: `python train_cv_test_report.py`).
