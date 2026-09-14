# Qualidade de dados e sobreposição entre amostras

## 1. Objetivo

Este documento registra a análise de qualidade de dados realizada após a auditoria técnica do projeto MedTriage.

O objetivo é quantificar repetições de texto, sobreposição entre conjuntos e ocorrência de múltiplos rótulos associados ao mesmo texto, preservando integralmente a avaliação oficial já utilizada no projeto.

Esta análise é complementar. Ela não substitui os resultados oficiais do baseline e não altera o pipeline de treinamento existente.

## 2. Metodologia

O diagnóstico reutiliza exatamente o mesmo fluxo de preparação de dados do projeto:

```text
medical_tc_train.csv
        ↓
load_dataset()
        ↓
add_triage_labels()
        ↓
normalização de texto
(strip + colapso de espaços)
        ↓
split_training_data()
        ↓
effective_train + validation
```

O teste oficial é carregado e preparado com o mesmo procedimento de normalização e mapeamento de `triage_label`.

A divisão treino/validação preserva:

```text
random_state = 837
stratify = triage_label
validation_size = configuração atual do projeto
```

O diagnóstico não aplica lowercase adicional, stemming, lematização ou outra transformação textual além da já existente no pipeline.

## 3. Artefato reproduzível

A análise é implementada em:

```text
src/medtriage/data/diagnostics.py
```

Testes unitários:

```text
tests/unit/test_data_diagnostics.py
```

Artefato produzido:

```text
artifacts/evaluation/data_overlap_analysis.json
```

Execução:

```bash
poetry run python -m medtriage.data.diagnostics
```

## 4. Estatísticas dos conjuntos

### Official train antes do split

```text
rows                       = 11550
unique texts               = 9445
rows in duplicated groups  = 4061
duplicates after first     = 2105
```

### Effective train

```text
rows                       = 9240
unique texts               = 7892
rows in duplicated groups  = 2614
duplicates after first     = 1348
```

### Validation

```text
rows                       = 2310
unique texts               = 2212
rows in duplicated groups  = 196
duplicates after first     = 98
```

### Official test

```text
rows                       = 2888
unique texts               = 2770
rows in duplicated groups  = 231
duplicates after first     = 118
```

### Full corpus

```text
rows                       = 14438
unique texts               = 11227
rows in duplicated groups  = 6140
duplicates after first     = 3211
```

## 5. Sobreposição entre conjuntos

### Effective train -> validation

```text
validation rows whose text exists in effective train = 677
shared unique texts                                   = 659
```

### Effective train -> official test

```text
official test rows whose text exists in effective train = 846
shared unique texts                                      = 828
```

### Validation -> official test

```text
official test rows whose text exists in validation = 199
shared unique texts                                 = 195
```

Esses valores demonstram que parte das amostras de validação e teste contém textos também presentes em outros conjuntos.

A contagem de linhas afetadas é diferente da contagem de textos únicos compartilhados porque um mesmo texto pode aparecer mais de uma vez.

## 6. Múltiplos condition_label por texto

### Official train

```text
texts with multiple condition_label = 1956
rows involved                       = 4061
```

Distribuição da quantidade de labels distintas por texto:

```text
2 labels -> 1810 textos
3 labels -> 143 textos
4 labels -> 3 textos
```

### Full corpus

```text
texts with multiple condition_label = 2929
rows involved                       = 6140
```

Distribuição:

```text
2 labels -> 2653 textos
3 labels -> 270 textos
4 labels -> 6 textos
```

## 7. Múltiplos triage_label por texto

### Official train

```text
texts with multiple triage_label = 1880
rows involved                    = 3909
```

Distribuição:

```text
2 labels -> 1760 textos
3 labels -> 120 textos
```

### Full corpus

```text
texts with multiple triage_label = 2837
rows involved                    = 5956
```

Distribuição:

```text
2 labels -> 2620 textos
3 labels -> 217 textos
```

## 8. Interpretação

A análise confirma que o corpus contém:

- textos repetidos;
- repetições dentro de um mesmo conjunto;
- textos compartilhados entre treino, validação e teste;
- textos idênticos associados a múltiplos `condition_label`;
- textos idênticos associados a múltiplos `triage_label`.

Essas características reduzem a independência estatística entre algumas amostras e exigem cautela na interpretação das métricas.

Entretanto, os resultados desta análise não são suficientes para concluir, de forma categórica, que houve inflação das métricas do modelo.

A auditoria original também observou que o subconjunto de textos repetidos não apresentou desempenho superior ao restante, portanto não existe evidência suficiente para afirmar que a sobreposição beneficiou artificialmente o modelo.

## 9. Impacto sobre a avaliação oficial

Os resultados oficiais do baseline são preservados.

O pipeline atual continua utilizando:

```text
split estratificado
random seed = 837
triage_label como target acadêmico
teste oficial do dataset preservado
```

Nenhuma amostra foi removida seletivamente e nenhum rótulo foi escolhido ou alterado para melhorar métricas.

A análise de qualidade deve ser tratada como contexto adicional para interpretar os resultados do Tech Challenge.

## 10. Sobre a proxy acadêmica

O campo `triage_label` continua sendo uma simplificação acadêmica criada a partir de `condition_label`.

Esse mapeamento existe para viabilizar a demonstração de uma arquitetura de Machine Learning e MLOps e não representa protocolo clínico real ou validação médica.

## 11. Limitações do diagnóstico

O diagnóstico atual:

- utiliza igualdade de texto após a normalização já existente no projeto;
- não aplica similaridade semântica ou fuzzy matching;
- não tenta determinar qual rótulo seria semanticamente correto;
- não remove duplicatas;
- não modifica o split oficial;
- não recalcula as métricas do baseline;
- não estima causalmente o impacto da sobreposição sobre desempenho.

Portanto, ele mede dependência observável por igualdade textual, mas não resolve ambiguidades presentes no corpus.

## 12. Validação complementar agrupada por texto

Uma possível evolução é criar uma avaliação complementar em que textos idênticos sejam mantidos no mesmo grupo durante o split, impedindo que o mesmo texto apareça simultaneamente em treino e validação.

Essa estratégia pode ser útil para medir a sensibilidade das métricas à dependência entre amostras.

Caso implementada, essa avaliação deve:

- utilizar o texto normalizado como identificador de grupo;
- impedir compartilhamento do mesmo texto entre treino e validação;
- ser apresentada como análise complementar;
- preservar os resultados oficiais atuais;
- não substituir silenciosamente a metodologia original.

Essa validação agrupada não foi adotada como avaliação principal neste estágio.

## 13. Conclusão

O diagnóstico pós-auditoria reproduziu os principais achados de qualidade de dados de forma automatizada e testável.

Os números confirmam presença relevante de duplicação, sobreposição entre conjuntos e múltiplos rótulos para textos idênticos.

A conclusão correta é que as métricas do baseline devem ser interpretadas com cautela devido à redução de independência entre algumas amostras.

Não existe evidência suficiente para afirmar que houve inflação das métricas, e os resultados oficiais permanecem preservados como baseline principal do projeto.