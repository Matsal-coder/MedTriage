# Resultados Baseline — MedTriage MLOps

## 1. Objetivo

Este documento consolida os resultados do modelo baseline implementado no Bloco 2 do Tech Challenge.

O objetivo do baseline é estabelecer uma referência reproduzível para:

- qualidade preditiva;
- comportamento por classe;
- persistência;
- inferência;
- latência;
- comparação futura com uma versão otimizada em ONNX.

IMPORTANTE:
As classes `normal`, `attention` e `urgent` são uma proxy acadêmica derivada das classes originais do Medical Abstracts TC Corpus. Esses resultados não representam desempenho clínico validado.

## 2. Dataset

Dataset:

```text
Medical Abstracts Text Classification Corpus
```

Arquivos:

```text
medical_tc_train.csv
medical_tc_test.csv
medical_tc_labels.csv
```

Amostras:

```text
train oficial = 11.550
test oficial  = 2.888
total         = 14.438
```

Colunas principais:

```text
medical_abstract
condition_label
```

Coluna adicionada:

```text
triage_label
```

## 3. Classes originais

```text
1 -> Neoplasms
2 -> Digestive system diseases
3 -> Nervous system diseases
4 -> Cardiovascular diseases
5 -> General pathological conditions
```

## 4. Mapeamento acadêmico

```text
Neoplasms                       -> urgent
Digestive system diseases       -> attention
Nervous system diseases         -> attention
Cardiovascular diseases         -> urgent
General pathological conditions -> normal
```

Esse agrupamento é uma simplificação acadêmica.

## 5. Split

O teste oficial é preservado integralmente.

O conjunto oficial de treino é dividido em:

```text
train      = 80%
validation = 20%
```

com:

```text
random seed = 837
```

e estratificação por `triage_label`.

Tamanho da validação:

```text
2310
```

Tamanho do teste:

```text
2888
```

## 6. Modelo

Pipeline:

```text
TfidfVectorizer
+
LogisticRegression
```

Parâmetros:

```text
TF-IDF:
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
lowercase = True

Logistic Regression:
max_iter = 1000
random_state = 837
```

Persistência:

```text
artifacts/models/baseline_pipeline.joblib
```

## 7. Métricas utilizadas

Foram calculadas:

- accuracy;
- precision macro;
- recall macro;
- F1 macro;
- F1 weighted;
- precision por classe;
- recall por classe;
- F1 por classe;
- support por classe;
- recall da classe urgent;
- matriz de confusão.

## 8. Resultados — Validação

```text
samples          : 2310
accuracy         : 0.5632
precision_macro  : 0.5351
recall_macro     : 0.5273
f1_macro         : 0.5284
f1_weighted      : 0.5558
urgent_recall    : 0.7505
```

### Por classe

```text
attention
precision : 0.4815
recall    : 0.3803
f1        : 0.4249
support   : 547

normal
precision : 0.4554
recall    : 0.4512
f1        : 0.4533
support   : 769

urgent
precision : 0.6685
recall    : 0.7505
f1        : 0.7071
support   : 994
```

### Matriz de confusão

Ordem:

```text
attention
normal
urgent
```

Matriz:

```text
[208, 230, 109]
[161, 347, 261]
[ 63, 185, 746]
```

## 9. Resultados — Teste

```text
samples          : 2888
accuracy         : 0.5765
precision_macro  : 0.5524
recall_macro     : 0.5447
f1_macro         : 0.5457
f1_weighted      : 0.5684
urgent_recall    : 0.7675
```

### Por classe

```text
attention
precision : 0.5281
recall    : 0.4401
f1        : 0.4801
support   : 684

normal
precision : 0.4638
recall    : 0.4266
f1        : 0.4444
support   : 961

urgent
precision : 0.6653
recall    : 0.7675
f1        : 0.7127
support   : 1243
```

### Matriz de confusão

Ordem:

```text
attention
normal
urgent
```

Matriz:

```text
[301, 254, 129]
[200, 410, 351]
[ 69, 220, 954]
```

## 10. Interpretação

Os resultados de validação e teste são próximos.

```text
validation f1_macro = 0.5284
test f1_macro       = 0.5457
```

Não existe diferença excessiva entre os conjuntos.

A classe de melhor desempenho é `urgent`.

No teste:

```text
urgent precision = 0.6653
urgent recall    = 0.7675
urgent f1        = 0.7127
```

A classe `attention` apresenta maior dificuldade de separação.

Essa dificuldade é coerente com a simplificação adotada, pois categorias médicas semanticamente distintas foram agrupadas em uma mesma classe acadêmica.

## 11. Limitação principal do target

O corpus não contém urgência clínica real.

O projeto converte:

```text
categorias médicas
        ↓
proxy acadêmica de triagem
```

Isso significa que o modelo está aprendendo principalmente padrões associados às categorias do corpus e ao agrupamento definido no projeto.

Por esse motivo, as métricas não devem ser interpretadas como desempenho de um sistema de triagem médica real.

## 12. Benchmark baseline de latência

Escopo:

```text
model_inference
```

O benchmark mede o `PredictionService` com o modelo previamente carregado em memória.

Não mede latência HTTP end-to-end.

## 13. Metodologia de benchmark

```text
modelo carregado uma vez
        ↓
20 inferências de warm-up
        ↓
500 inferências medidas
        ↓
5 textos fixos
        ↓
repetição cíclica dos inputs
```

Métricas:

```text
mean
p50
p95
min
max
throughput
```

## 14. Benchmark histórico do Bloco 2

A primeira medição de latência do baseline foi realizada durante o Bloco 2,
antes da etapa final de otimização e comparação com ONNX.

Resultados históricos observados:

```text
mean       : 2.2997 ms
p50        : 2.2074 ms
p95        : 2.8114 ms
min        : 1.8789 ms
max        : 4.0201 ms
throughput : 434.84 req/s
```

Esses valores são preservados como registro histórico do desenvolvimento e
não devem ser usados como referência para a comparação final com ONNX.

## 15. Ambiente do benchmark histórico

```text
Python:
3.12.2

Platform:
Windows-10-10.0.19045-SP0

Processor:
Intel64 Family 6 Model 69 Stepping 1, GenuineIntel
```

## 16. Artefatos de benchmark

O caminho utilizado pelo projeto é:

```text
artifacts/benchmarks/baseline_latency.json
```

Durante o Bloco 2, esse caminho foi utilizado para persistir a medição
histórica do baseline.

Na etapa final do Bloco 5, o benchmark foi executado novamente no contexto da
comparação com ONNX. O arquivo `baseline_latency.json` atualmente versionado
corresponde a essa medição comparativa final, e não aos números históricos
registrados na seção anterior.

Os artefatos finais versionados são:

```text
artifacts/benchmarks/baseline_latency.json
artifacts/benchmarks/optimized_latency.json
artifacts/benchmarks/latency_comparison.json
```

## 17. Relação com a API

O endpoint `/predict` também retorna:

```text
inference_time_ms
```

Esse valor representa a latência observada em uma chamada individual.

Exemplo histórico observado durante smoke test local:

```text
18.1404 ms
```

Uma medição isolada não deve ser comparada diretamente com média, p50 ou p95
de um benchmark agregado com warm-up e centenas de execuções.

## 18. Referência final para comparação com ONNX

A comparação oficial final foi produzida no Bloco 5, executando baseline e
ONNX na mesma sessão de benchmark, com os mesmos textos, quantidade de
warm-up e número de medições.

A referência final do baseline é:

```text
MODEL
TF-IDF + LogisticRegression

QUALITY
test f1_macro      = 0.5457
test urgent_recall = 0.7675

FINAL LATENCY REFERENCE
mean       = 2.958672 ms
p50        = 2.702200 ms
p95        = 3.849020 ms
throughput = 337.989522 req/s
```

A fonte normativa para a comparação final de latência é:

```text
artifacts/benchmarks/baseline_latency.json
artifacts/benchmarks/optimized_latency.json
artifacts/benchmarks/latency_comparison.json
```

## 19. Regras da comparação final

Baseline e ONNX foram comparados preservando:

- os mesmos 5 textos;
- 20 execuções de warm-up por backend;
- 500 medições por backend;
- o mesmo escopo `model_inference`;
- a mesma máquina e sessão de trabalho;
- modelos previamente carregados.

A execução foi **sequencial**: primeiro o benchmark de um backend e depois o
do outro. Não houve alternância A/B nem randomização da ordem.

A comparação mede os serviços de inferência implementados e não isola
perfeitamente o custo puro dos runtimes.


## 20. Conclusão

O baseline cumpre o papel arquitetural esperado para o Bloco 2:

- modelo treinável;
- pipeline reproduzível;
- artefato persistido;
- avaliação objetiva;
- inferência real;
- latência mensurável;
- integração com FastAPI;
- integração com Docker;
- referência preparada para futura otimização.