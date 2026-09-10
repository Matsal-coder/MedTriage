# Arquitetura — MedTriage MLOps

## 1. Objetivo do sistema

O MedTriage MLOps é um projeto acadêmico voltado à construção de uma solução de classificação de urgência em textos médicos.

O objetivo final é receber um texto ou laudo e classificá-lo em uma categoria acadêmica de triagem:

- `normal`;
- `attention`;
- `urgent`.

IMPORTANTE:
Essas classes são uma proxy criada para o Tech Challenge a partir das categorias originais do Medical Abstracts TC Corpus. O sistema não representa uma solução clínica validada.

A solução completa contempla não apenas Machine Learning, mas também API, containerização, automação, observabilidade e otimização de inferência.

## 2. Estado atual da arquitetura

Ao final do BLOCO 2, a arquitetura real implementada é:

```text
Medical Abstracts TC Corpus
          ↓
      data/loader.py
          ↓
   data/validation.py
          ↓
   target acadêmico
          ↓
 train / validation / test
          ↓
 TfidfVectorizer
          +
 LogisticRegression
          ↓
 baseline_pipeline.joblib
      ↙            ↘
evaluate.py      predict.py
    ↓                ↓
evaluation.json   FastAPI
                    ↓
               POST /predict
                    ↓
               Docker runtime

baseline_pipeline.joblib
          ↓
benchmarking/latency.py
          ↓
baseline_latency.json
```

O `GET /health` herdado do Bloco 1 permanece preservado.

## 3. Arquitetura alvo do projeto

```text
Dataset
 ↓
Airflow
 ↓
Treinamento
 ↓
Modelo
 ↓
Scikit-learn / ONNX
 ↓
FastAPI
 ↓
Docker
 ↓
/predict | /health | /metrics
                      ↓
                  Prometheus
                      ↓
                    Grafana
```

GitHub Actions atuará transversalmente:

```text
Git push / Pull Request
          ↓
     GitHub Actions
          ↓
        Ruff
          ↓
        pytest
          ↓
     Docker build
```

Os componentes ainda não implementados permanecem reservados aos blocos seguintes.

## 4. Dataset e camada de dados

### Dataset

Foi adotado o Medical Abstracts Text Classification Corpus.

Arquivos principais:

```text
medical_tc_train.csv
medical_tc_test.csv
medical_tc_labels.csv
```

Os arquivos são mantidos localmente em:

```text
data/raw/
```

e ignorados pelo Git.

### Contrato original

Colunas utilizadas:

```text
medical_abstract
condition_label
```

Labels originais:

```text
1 -> Neoplasms
2 -> Digestive system diseases
3 -> Nervous system diseases
4 -> Cardiovascular diseases
5 -> General pathological conditions
```

### Proxy acadêmica

O projeto adiciona:

```text
triage_label
```

através do mapeamento:

```text
1 -> urgent
2 -> attention
3 -> attention
4 -> urgent
5 -> normal
```

`condition_label` é preservado.

Esse mapeamento é uma simplificação acadêmica e não um protocolo clínico.

## 5. `src/medtriage/data/loader.py`

Responsabilidades:

- carregar CSV;
- chamar validação;
- preservar colunas originais;
- normalizar whitespace no texto;
- adicionar `triage_label`;
- realizar split reproduzível.

Funções principais:

```text
load_dataset()
add_triage_labels()
split_training_data()
```

## 6. `src/medtriage/data/validation.py`

Responsabilidades:

- rejeitar dataset vazio;
- validar colunas obrigatórias;
- rejeitar nulls;
- rejeitar abstracts vazios;
- validar labels originais.

O objetivo é manter validação separada de transformação.

## 7. Split dos dados

O corpus possui:

```text
11.550 treino oficial
2.888 teste oficial
```

O treino oficial é dividido em:

```text
80% treino
20% validação
```

com:

```text
random_state = 837
```

e estratificação por `triage_label`.

O conjunto de teste oficial permanece reservado para avaliação final.

## 8. Configuração centralizada

`src/medtriage/config.py` concentra os valores globais.

Principais grupos:

### Aplicação

```text
APP_NAME
APP_VERSION
DEFAULT_LOG_LEVEL
```

### Reprodutibilidade

```text
RANDOM_SEED = 837
VALIDATION_SIZE = 0.20
```

### Dados

```text
DATA_DIR
RAW_DATA_DIR
PROCESSED_DATA_DIR
TRAIN_DATA_PATH
TEST_DATA_PATH
TEXT_COLUMN
ORIGINAL_TARGET_COLUMN
TRIAGE_TARGET_COLUMN
```

### Artefatos

```text
ARTIFACTS_DIR
MODELS_DIR
MODEL_ARTIFACT_PATH
EVALUATION_ARTIFACT_PATH
BENCHMARKS_DIR
BASELINE_BENCHMARK_PATH
```

### Modelo

```text
TFIDF_NGRAM_RANGE
TFIDF_MIN_DF
TFIDF_MAX_DF
LOGISTIC_REGRESSION_MAX_ITER
TRIAGE_CLASSES
```

### Benchmark

```text
BENCHMARK_WARMUP_RUNS
BENCHMARK_MEASURED_RUNS
```

A centralização evita duplicação de seed, paths, nomes de colunas e nomes de artefatos.

## 9. Modelo baseline

O baseline utiliza:

```text
TfidfVectorizer
+
LogisticRegression
```

dentro de:

```text
sklearn.pipeline.Pipeline
```

Parâmetros:

```text
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
lowercase = True
max_iter = 1000
random_state = 837
```

O pipeline único reduz risco de inconsistência entre treino e inferência.

## 10. `src/medtriage/modeling/train.py`

Responsabilidades:

- construir pipeline;
- treinar modelo;
- persistir artefato;
- oferecer função reutilizável de treinamento;
- expor comando CLI.

Funções:

```text
build_model_pipeline()
train_model()
persist_model()
run_training()
main()
```

Fluxo:

```text
medical_tc_train.csv
 ↓
load_dataset
 ↓
add_triage_labels
 ↓
split_training_data
 ↓
train_model
 ↓
persist_model
 ↓
baseline_pipeline.joblib
```

O `run_training()` foi mantido reutilizável para que o Airflow no Bloco 3 possa orquestrar o treino sem duplicar lógica dentro da DAG.

## 11. Persistência do modelo

Artefato principal:

```text
artifacts/models/baseline_pipeline.joblib
```

Formato:

```text
Joblib
```

Conteúdo:

```text
TF-IDF
+
LogisticRegression
```

persistidos juntos.

Vantagens:

- contrato único;
- menor risco de incompatibilidade;
- carregamento simples;
- facilidade de uso na API;
- futura referência para conversão/otimização.

## 12. `src/medtriage/modeling/evaluate.py`

Responsabilidades:

- carregar modelo persistido;
- reconstruir validation set reproduzível;
- carregar test set oficial;
- calcular métricas;
- persistir avaliação.

Fluxo:

```text
baseline_pipeline.joblib
      ↓
validation set
      +
test set
      ↓
evaluate_model()
      ↓
calculate_metrics()
      ↓
evaluation.json
```

Métricas:

- accuracy;
- precision macro;
- recall macro;
- F1 macro;
- F1 weighted;
- métricas por classe;
- recall de `urgent`;
- matriz de confusão.

Artefato:

```text
artifacts/models/evaluation.json
```

## 13. Resultados do modelo

### Validação

```text
samples           2310
accuracy          0.5632
precision_macro   0.5351
recall_macro      0.5273
f1_macro          0.5284
f1_weighted       0.5558
urgent_recall     0.7505
```

### Teste

```text
samples           2888
accuracy          0.5765
precision_macro   0.5524
recall_macro      0.5447
f1_macro          0.5457
f1_weighted       0.5684
urgent_recall     0.7675
```

Os resultados próximos entre validação e teste indicam comportamento consistente entre os conjuntos.

## 14. `src/medtriage/modeling/predict.py`

`PredictionService` encapsula:

- caminho do modelo;
- carregamento;
- inferência;
- probabilidades;
- medição de tempo.

Fluxo:

```text
PredictionService()
 ↓
load()
 ↓
joblib.load()
 ↓
modelo em memória
 ↓
predict(text)
```

O artefato não é recarregado a cada request.

## 15. API FastAPI

Entry point preservado:

```text
medtriage.api.app:app
```

### Lifespan

Fluxo:

```text
FastAPI startup
 ↓
create_prediction_service()
 ↓
PredictionService.load()
 ↓
modelo residente em memória
```

Essa estratégia evita acesso a disco em cada request.

## 16. `GET /health`

Contrato preservado:

```text
GET /health
```

Resposta mínima:

```json
{"status":"ok"}
```

## 17. `POST /predict`

Request:

```json
{
  "text": "Patient presents with..."
}
```

Response:

```json
{
  "prediction": "urgent",
  "probabilities": {
    "attention": 0.1,
    "normal": 0.2,
    "urgent": 0.7
  },
  "inference_time_ms": 2.0
}
```

Entradas vazias ou apenas whitespace são rejeitadas com HTTP 422.

## 18. Schemas

`src/medtriage/api/schemas.py` contém:

```text
HealthResponse
PredictionRequest
PredictionResponse
```

Os contratos HTTP permanecem separados da lógica de ML.

## 19. Benchmark de latência

Arquivo:

```text
src/medtriage/benchmarking/latency.py
```

Responsabilidades:

- carregar `PredictionService`;
- warm-up;
- inferências medidas;
- estatísticas;
- persistência.

Metodologia:

```text
20 warm-ups
500 medições
5 inputs fixos
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

Resultado obtido:

```text
mean       2.2997 ms
p50        2.2074 ms
p95        2.8114 ms
min        1.8789 ms
max        4.0201 ms
throughput 434.84 req/s
```

Escopo:

```text
model_inference
```

Portanto, não representa latência HTTP end-to-end.

Artefato:

```text
artifacts/benchmarks/baseline_latency.json
```

## 20. Docker

Base:

```text
python:3.12.2-slim
```

Usuário:

```text
appuser
```

Arquivos necessários ao runtime:

```text
src/
artifacts/models/baseline_pipeline.joblib
```

O modelo é incorporado durante o `docker build`.

Fluxo:

```text
artefato local
 ↓
docker build
 ↓
COPY baseline_pipeline.joblib
 ↓
imagem Docker
 ↓
Uvicorn
 ↓
FastAPI lifespan
 ↓
PredictionService.load()
```

Smoke tests validados:

```text
GET /health
POST /predict
docker exec ... whoami
presença do joblib dentro do container
```

O container continua rodando sem privilégios de root.

## 21. Artefatos

```text
artifacts/
├── models/
│   ├── baseline_pipeline.joblib
│   └── evaluation.json
└── benchmarks/
    └── baseline_latency.json
```

Eles são ignorados pelo Git.

## 22. Testes

A suíte cobre:

### Dados

- dataset válido;
- dataset vazio;
- coluna ausente;
- texto nulo;
- texto vazio;
- label inválido;
- pipeline de preparação;
- reprodutibilidade do split.

### Treinamento

- construção do pipeline;
- treinamento;
- persistência;
- reload;
- pipeline completo.

### Avaliação

- contrato de métricas;
- scores perfeitos;
- persistência;
- workflow completo.

### Predição

- carregamento;
- predição;
- probabilidades;
- ausência de artefato;
- modelo não carregado.

### API

- import;
- `/health`;
- `/predict`;
- entradas inválidas.

### Benchmark

- percentile;
- estatísticas;
- erros para dados vazios;
- persistência.

## 23. Fluxo de treinamento

```text
Medical Abstracts train CSV
          ↓
       loader
          ↓
      validation
          ↓
 target acadêmico
          ↓
 split reproduzível
          ↓
      train data
          ↓
       TF-IDF
          ↓
Logistic Regression
          ↓
baseline_pipeline.joblib
```

## 24. Fluxo de avaliação

```text
baseline_pipeline.joblib
        ↓
validation set + test set
        ↓
predict
        ↓
metrics
        ↓
evaluation.json
```

## 25. Fluxo de inferência

```text
Cliente
 ↓
HTTP
 ↓
FastAPI
 ↓
PredictionService
 ↓
pipeline carregado em memória
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
prediction
 ↓
probabilities
 ↓
inference_time_ms
 ↓
Response
```

## 26. Integração futura com Airflow

O Bloco 3 não deve reimplementar treinamento dentro da DAG.

A DAG deve reutilizar funções existentes, especialmente:

```text
run_training()
run_evaluation()
```

A responsabilidade do Airflow será orquestrar essas etapas, não conter a lógica de modelagem.

## 27. Integração futura com GitHub Actions

O CI deverá executar pelo menos:

```bash
poetry install
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
docker build ...
```

A estratégia concreta para disponibilizar o modelo antes do build deverá ser definida no Bloco 3.

## 28. Integração futura com ONNX

O Bloco 5 deverá preservar, para comparação justa:

- mesmos inputs;
- mesmo tipo de latência;
- mesmo warm-up;
- mesmo número de execuções;
- mesmo ambiente sempre que possível.

Baseline oficial:

```text
mean       2.2997 ms
p50        2.2074 ms
p95        2.8114 ms
throughput 434.84 req/s
```

O contrato HTTP idealmente deve permanecer estável mesmo que o backend de inferência seja trocado.

## 29. Estratégia de cloud

Google Cloud Run permanece como referência arquitetural teórica.

A aplicação não depende especificamente do Cloud Run e pode ser executada em serviços equivalentes.

Não existe deploy real em cloud no Bloco 2.

## 30. Limites atuais

Ainda não existem:

```text
Airflow
DAG
GitHub Actions
Prometheus
Grafana
/metrics
ONNX
quantização
benchmark comparativo
vídeo STAR
```

Essas ausências são deliberadas e pertencem aos próximos blocos.