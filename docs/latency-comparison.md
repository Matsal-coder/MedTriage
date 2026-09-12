# Comparação de Latência — Baseline vs ONNX

## 1. Objetivo

Este documento registra a técnica de otimização aplicada ao MedTriage e a comparação de latência entre:

```text
Baseline:
TfidfVectorizer sklearn
+
LogisticRegression sklearn
```

e:

```text
Otimizado:
TfidfVectorizer sklearn
+
LogisticRegression ONNX Runtime
```

A comparação foi construída com prioridade para:

1. corretude;
2. equivalência funcional;
3. metodologia justa;
4. reprodutibilidade;
5. transparência sobre os resultados.

## 2. Baseline

O modelo original é uma `sklearn.pipeline.Pipeline`:

```text
texto
 ↓
TfidfVectorizer
 ↓
LogisticRegression
 ↓
classe + probabilidades
```

Artefato:

```text
artifacts/models/baseline_pipeline.joblib
```

O vectorizer real possui:

```text
186.957 features
```

## 3. Primeira estratégia ONNX

A primeira tentativa converteu toda a pipeline:

```text
texto
 ↓
TF-IDF ONNX
 ↓
LogisticRegression ONNX
```

O modelo convertido:

- foi criado com sucesso;
- passou pelo `onnx.checker`;
- foi carregado pelo ONNX Runtime;
- utilizou `CPUExecutionProvider`;
- produziu classes coerentes.

Entretanto, os testes de equivalência mostraram divergência significativa de probabilidades.

Exemplos observados:

```text
diferença máxima ≈ 0.0373
diferença máxima ≈ 0.0689
diferença máxima ≈ 0.0514
```

A maior diferença observada foi aproximadamente:

```text
0.0689
```

Embora as classes previstas fossem iguais nos textos testados, essa diferença não foi considerada aceitável.

A tolerância não foi ampliada artificialmente.

## 4. Decisão arquitetural

A investigação indicou que a origem provável da diferença estava na reprodução do preprocessing textual durante a conversão ONNX.

A estratégia final passou a preservar o `TfidfVectorizer` do sklearn e converter somente a regressão logística.

Arquitetura final:

```text
texto
 ↓
TfidfVectorizer sklearn
 ↓
CSR float64
 ↓
astype(float32)
 ↓
toarray()
 ↓
LogisticRegression ONNX
 ↓
prediction + probabilities
```

## 5. Modelo ONNX final

Input do modelo real:

```text
features
shape: [None, 186957]
type: tensor(float)
```

Outputs:

```text
label
shape: [None]
type: tensor(string)
```

```text
probabilities
shape: [None, 3]
type: tensor(float)
```

Provider:

```text
CPUExecutionProvider
```

## 6. Equivalência funcional

A versão híbrida foi submetida a testes automatizados.

Critérios:

```text
mesma classe prevista
mesmas labels
mesma ordem de classes
probabilidades próximas
```

Tolerância:

```text
absolute tolerance = 1e-5
relative tolerance = 0
```

Resultado:

```text
19/19 testes de equivalência aprovados
```

A versão otimizada só foi considerada válida depois dessa etapa.

## 7. Metodologia do benchmark

Os dois backends utilizam:

```text
mesmos 5 textos
20 warm-ups
500 medições
mesma máquina
mesmo escopo
modelos carregados previamente
```

Escopo:

```text
model_inference
```

Textos:

```text
1. Patient with cardiovascular symptoms and chest pain.
2. Patient undergoing evaluation for digestive system symptoms.
3. General pathological condition under routine clinical assessment.
4. Patient with neurological symptoms requiring medical evaluation.
5. Patient with suspected neoplasm undergoing diagnostic investigation.
```

A comparação final executa baseline e ONNX na mesma sessão de trabalho para reduzir diferenças causadas pelo ambiente.

## 8. O que entra na medição

### Baseline

O caminho medido corresponde ao comportamento real de `PredictionService`.

### Backend ONNX

O cronômetro inclui:

```text
TF-IDF sklearn
 ↓
conversão CSR para float32
 ↓
conversão para tensor denso
 ↓
ONNX Runtime
```

A conversão para denso não foi retirada da medição porque faz parte do caminho real de inferência otimizado.

## 9. Resultados pareados finais

### Baseline

```text
mean       2.958672 ms
p50        2.702200 ms
p95        3.849020 ms
min        2.388500 ms
max        6.718400 ms
throughput 337.989522 req/s
```

### ONNX

```text
mean       2.860280 ms
p50        2.393200 ms
p95        5.123730 ms
min        1.368800 ms
max        19.612500 ms
throughput 349.616121 req/s
```

## 10. Tabela comparativa

| Métrica | Baseline | ONNX | Interpretação |
|---|---:|---:|---|
| mean | 2.9587 ms | 2.8603 ms | ONNX levemente melhor |
| p50 | 2.7022 ms | 2.3932 ms | ONNX melhor |
| p95 | 3.8490 ms | 5.1237 ms | ONNX pior |
| min | 2.3885 ms | 1.3688 ms | ONNX melhor no mínimo |
| max | 6.7184 ms | 19.6125 ms | maior variabilidade no ONNX |
| throughput | 337.99 req/s | 349.62 req/s | ONNX levemente melhor |

## 11. Speedup

Fórmula:

```text
speedup = baseline_latency / optimized_latency
```

Resultado:

```text
mean_speedup = 1.0343993x
p50_speedup  = 1.1291158x
p95_speedup  = 0.7512144x
```

Interpretação:

```text
speedup > 1.0 -> otimizado mais rápido
speedup = 1.0 -> equivalente
speedup < 1.0 -> otimizado mais lento
```

Portanto:

- média: pequena melhora no ONNX;
- mediana: melhora no ONNX;
- p95: regressão significativa;
- throughput: pequena melhora;
- latência de cauda: pior e mais variável no ONNX.

## 12. Conclusão

O backend ONNX preservou equivalência funcional e apresentou ganho em média e p50 nesta execução pareada, porém não apresentou ganho consistente em toda a distribuição de latência. O p95 piorou de forma relevante e o backend ONNX mostrou maior variabilidade.

Isso é compatível com as características do modelo:

- `LogisticRegression` possui inferência barata;
- o TF-IDF possui 186.957 features;
- sklearn trabalha naturalmente com representação esparsa;
- ONNX recebe tensor `float32` denso;
- a conversão CSR -> dense adiciona overhead.

Assim:

```text
ganho potencial no classificador
-
custo de preparação do tensor
=
benefício variável conforme a métrica observada
```

Como não houve ganho consistente suficiente para justificar a troca do backend padrão, a API final permanece utilizando sklearn. O ONNX é mantido como alternativa validada, entregue e benchmarkada.

## 13. Por que o resultado não foi alterado

O objetivo do benchmark é avaliar a técnica de otimização, não demonstrar obrigatoriamente ganho.

Não foram adotadas as seguintes práticas:

- excluir outliers sem critério;
- medir somente a parte ONNX e ignorar preprocessing;
- comparar máquinas diferentes;
- alterar os textos;
- reduzir medições;
- aumentar tolerância de equivalência;
- reportar somente a métrica favorável.

O resultado final foi preservado integralmente: favorável ao ONNX em média/p50, mas desfavorável em p95 e variabilidade.

## 14. Artefatos

Como evidência da entrega final, são versionados:

```text
artifacts/models/optimized_model.onnx
artifacts/benchmarks/baseline_latency.json
artifacts/benchmarks/optimized_latency.json
artifacts/benchmarks/latency_comparison.json
```

O baseline sklearn, `evaluation.json` e demais artefatos intermediários continuam gerados localmente e ignorados pelo Git.

Código:

```text
src/medtriage/benchmarking/latency.py
src/medtriage/benchmarking/comparison.py
```

## 15. Reproduzir o benchmark

Primeiro gerar o artefato ONNX:

```bash
poetry run python -m medtriage.modeling.onnx_export
```

Depois:

```bash
poetry run python -m medtriage.benchmarking.comparison
```

O comando executa baseline e ONNX e gera a comparação.

## 16. Validação

Após as mudanças:

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
```

Estado observado:

```text
84 passed
Ruff check OK
Ruff format OK
```

## 17. Limitações

- benchmark executado em máquina local;
- resultados dependem de CPU, carga e sistema operacional;
- o modelo é simples e linear;
- não foi utilizada GPU;
- a alta dimensionalidade do TF-IDF prejudica o caminho denso;
- o artefato ONNX otimiza somente o classificador;
- o preprocessing continua no sklearn;
- resultados não devem ser generalizados para arquiteturas neurais ou outros modelos.

## 18. Resumo executivo

```text
Técnica aplicada: ONNX Runtime
Conversão final: LogisticRegression
Preprocessing: TfidfVectorizer sklearn
Equivalência: validada
Benchmark: 20 warm-ups + 500 medições + 5 textos
Mean baseline: 2.9587 ms
Mean ONNX: 2.8603 ms
Mean speedup: 1.0344x
p50 speedup: 1.1291x
p95 speedup: 0.7512x
Resultado: ganho em média/p50, regressão no p95
Conclusão: otimização funcionalmente correta, mas sem ganho consistente suficiente para substituir sklearn na API
```
