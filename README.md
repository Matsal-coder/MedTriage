# MedTriage MLOps

Projeto desenvolvido para o Tech Challenge da Fase 3 da Pós-Tech FIAP.

O objetivo geral é construir uma solução de MLOps para classificação acadêmica de urgência em textos médicos, incluindo API de inferência, containerização, automação de pipelines, observabilidade e otimização de latência.

IMPORTANTE:
O mapeamento para as classes `normal`, `attention` e `urgent` é uma simplificação acadêmica criada para demonstrar a arquitetura MLOps. Ele NÃO representa uma regra clínica validada e NÃO deve ser interpretado como sistema médico de triagem real.

## Status do projeto

O projeto está no final do BLOCO 2 — Dataset, NLP, treinamento, API real e latência baseline.

Neste estágio já estão implementados:

- estrutura Python em layout `src/`;
- gerenciamento de dependências com Poetry;
- lint e formatação com Ruff;
- testes com pytest;
- configuração centralizada;
- logging básico com a biblioteca padrão;
- aplicação FastAPI;
- endpoint `GET /health`;
- endpoint `POST /predict`;
- Dockerfile funcional;
- execução da API em container como usuário não-root;
- Medical Abstracts TC Corpus;
- validação e preparação dos dados;
- split reproduzível treino/validação;
- modelo baseline com TF-IDF + Logistic Regression;
- persistência do modelo com Joblib;
- avaliação em validação e teste;
- carregamento do modelo no startup da API;
- benchmark baseline de latência;
- documentação arquitetural atualizada.

Ainda não fazem parte do estado atual:

- GitHub Actions;
- Airflow;
- DAG de treinamento;
- Prometheus;
- Grafana;
- endpoint `/metrics`;
- ONNX;
- quantização;
- benchmark comparativo baseline vs modelo otimizado;
- vídeo STAR final.

## Stack

### Implementada

- Python 3.12.2
- Poetry
- FastAPI
- Uvicorn
- pandas
- scikit-learn
- Joblib
- pytest
- Ruff
- Docker

### Planejada para os próximos blocos

- GitHub Actions
- Apache Airflow
- Prometheus
- Grafana
- ONNX

## Estrutura atual

```text
medtriage-mlops/
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── docs/
│   ├── architecture.md
│   └── baseline-results.md
├── src/
│   └── medtriage/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   └── schemas.py
│       ├── benchmarking/
│       │   ├── __init__.py
│       │   └── latency.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── validation.py
│       └── modeling/
│           ├── __init__.py
│           ├── evaluate.py
│           ├── predict.py
│           └── train.py
├── tests/
│   ├── integration/
│   │   ├── test_data_pipeline.py
│   │   ├── test_evaluation_pipeline.py
│   │   ├── test_health.py
│   │   ├── test_predict.py
│   │   └── test_training_pipeline.py
│   └── unit/
│       ├── test_benchmarking.py
│       ├── test_config.py
│       ├── test_data_validation.py
│       ├── test_evaluation.py
│       ├── test_modeling.py
│       └── test_prediction.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Instalação

O projeto utiliza Poetry para gerenciamento de dependências.

```bash
poetry install
```

O pacote `medtriage` é instalado a partir de `src/medtriage/`.

## Qualidade e testes

```bash
poetry run pytest
```

```bash
poetry run ruff check .
```

```bash
poetry run ruff format --check .
```

## Dataset

O dataset adotado é o Medical Abstracts Text Classification Corpus.

Arquivos utilizados:

```text
medical_tc_train.csv
medical_tc_test.csv
medical_tc_labels.csv
```

Os arquivos completos do dataset ficam em:

```text
data/raw/
```

Eles NÃO são versionados no Git.

A estrutura dos diretórios é preservada no repositório através de arquivos `.gitkeep`.

## Colunas relevantes

O corpus utiliza:

```text
medical_abstract
condition_label
```

No projeto é adicionada também:

```text
triage_label
```

A coluna `condition_label` original é preservada para rastreabilidade.

## Mapeamento acadêmico para triagem

O corpus possui originalmente cinco categorias:

```text
1 -> Neoplasms
2 -> Digestive system diseases
3 -> Nervous system diseases
4 -> Cardiovascular diseases
5 -> General pathological conditions
```

Para adequar o projeto ao cenário acadêmico do Tech Challenge, foi adotado o seguinte mapeamento:

```text
1 -> urgent
2 -> attention
3 -> attention
4 -> urgent
5 -> normal
```

Esse mapeamento é deliberadamente simplificado e serve somente como proxy acadêmica.

Ele não representa conhecimento médico suficiente para inferir urgência clínica real.

## Pipeline de dados

Fluxo:

```text
CSV
 ↓
load_dataset()
 ↓
validate_dataset()
 ↓
add_triage_labels()
 ↓
split_training_data()
```

Validações implementadas:

- dataset não vazio;
- presença das colunas obrigatórias;
- ausência de abstracts nulos;
- ausência de labels nulos;
- abstracts não vazios;
- labels originais limitados a 1, 2, 3, 4 e 5.

O texto também passa por:

- `strip`;
- normalização de espaços em branco.

## Split

O corpus disponibiliza treino e teste oficiais.

O arquivo oficial de treino contém 11.550 amostras e é dividido em:

```text
80% treino
20% validação
```

com:

```text
RANDOM_SEED = 837
```

e estratificação por `triage_label`.

O arquivo oficial de teste possui 2.888 amostras e permanece reservado para avaliação final.

## Modelo baseline

O baseline é composto por:

```text
TfidfVectorizer
+
LogisticRegression
```

Parâmetros principais:

```text
TF-IDF
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
lowercase = True

Logistic Regression
max_iter = 1000
random_state = 837
```

O TF-IDF e o classificador ficam dentro de um único `sklearn.pipeline.Pipeline`.

## Treinamento

Execute:

```bash
poetry run python -m medtriage.modeling.train
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
TF-IDF + Logistic Regression
 ↓
fit
 ↓
baseline_pipeline.joblib
```

Artefato gerado:

```text
artifacts/models/baseline_pipeline.joblib
```

Esse artefato não é versionado no Git.

## Avaliação

Execute:

```bash
poetry run python -m medtriage.modeling.evaluate
```

Artefato gerado:

```text
artifacts/models/evaluation.json
```

As métricas incluem:

- accuracy;
- precision macro;
- recall macro;
- F1 macro;
- F1 weighted;
- métricas por classe;
- recall da classe `urgent`;
- matriz de confusão;
- quantidade de amostras avaliadas.

### Resultados — Validação

```text
samples           2310
accuracy          0.5632
precision_macro   0.5351
recall_macro      0.5273
f1_macro          0.5284
f1_weighted       0.5558
urgent_recall     0.7505
```

### Resultados — Teste

```text
samples           2888
accuracy          0.5765
precision_macro   0.5524
recall_macro      0.5447
f1_macro          0.5457
f1_weighted       0.5684
urgent_recall     0.7675
```

Os resultados de validação e teste são próximos, sem evidência de diferença excessiva entre os conjuntos.

## API local

Execute:

```bash
poetry run uvicorn medtriage.api.app:app --host 127.0.0.1 --port 8000
```

A API carrega o modelo uma única vez durante o lifespan do FastAPI.

Fluxo:

```text
FastAPI startup
 ↓
PredictionService
 ↓
joblib.load()
 ↓
baseline_pipeline.joblib em memória
 ↓
requests de inferência
```

## Health check

```bash
curl http://127.0.0.1:8000/health
```

Resposta:

```json
{"status":"ok"}
```

## Prediction

Request:

```bash
curl -X POST   http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"text":"Patient presents with acute cardiovascular symptoms and chest pain."}'
```

Contrato de resposta:

```json
{
  "prediction": "normal",
  "probabilities": {
    "attention": 0.3643997101916822,
    "normal": 0.4169552447148065,
    "urgent": 0.2186450450935112
  },
  "inference_time_ms": 18.14039999999295
}
```

Os valores exatos dependem da entrada e do ambiente.

Entradas vazias ou contendo somente whitespace retornam HTTP 422.

## Benchmark baseline de latência

Execute:

```bash
poetry run python -m medtriage.benchmarking.latency
```

Metodologia:

```text
carregamento único do modelo
 ↓
20 warm-ups
 ↓
500 inferências medidas
 ↓
5 textos fixos repetidos ciclicamente
 ↓
estatísticas agregadas
```

Resultado obtido no ambiente local:

```text
mean       2.2997 ms
p50        2.2074 ms
p95        2.8114 ms
min        1.8789 ms
max        4.0201 ms
throughput 434.84 req/s
```

Ambiente:

```text
Python 3.12.2
Windows 10
Intel64 Family 6 Model 69 Stepping 1, GenuineIntel
```

Esse benchmark mede somente inferência do modelo em memória.

Ele não representa latência HTTP end-to-end.

Artefato gerado:

```text
artifacts/benchmarks/baseline_latency.json
```

## Docker

A imagem utiliza:

```text
python:3.12.2-slim
```

e executa como:

```text
appuser
```

O runtime inclui:

```text
src/
artifacts/models/baseline_pipeline.joblib
```

Os artefatos de avaliação e benchmark não são necessários dentro da imagem.

Build:

```bash
docker build -t medtriage-mlops .
```

Execução:

```bash
docker run --rm -p 8000:8000 --name medtriage-api medtriage-mlops
```

Smoke tests:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST   http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"text":"Patient presents with acute cardiovascular symptoms and chest pain."}'
```

Verificação do usuário:

```bash
docker exec medtriage-api whoami
```

Esperado:

```text
appuser
```

Verificação do modelo dentro do container, usando Git Bash:

```bash
docker exec medtriage-api sh -c 'ls -lh /app/artifacts/models/'
```

ou:

```bash
docker exec medtriage-api sh -c 'test -f /app/artifacts/models/baseline_pipeline.joblib && echo MODEL_OK'
```

## Persistência de artefatos

```text
artifacts/
├── models/
│   ├── baseline_pipeline.joblib
│   └── evaluation.json
└── benchmarks/
    └── baseline_latency.json
```

Esses arquivos são gerados localmente e ignorados pelo Git.

## Decisão arquitetural de inferência

A principal forma de inferência é real-time.

Fluxo:

```text
Cliente
 ↓
POST /predict
 ↓
FastAPI
 ↓
PredictionService
 ↓
Pipeline TF-IDF + Logistic Regression
 ↓
prediction + probabilities + inference_time_ms
```

Batch continua sendo usado para:

- preparação de dados;
- treinamento;
- avaliação;
- benchmarks;
- futuro re-treinamento orquestrado.

## Estratégia de cloud

A arquitetura de referência continua considerando Google Cloud Run para uma eventual execução gerenciada do container.

Essa é uma decisão acadêmica e arquitetural.

Não existe deploy real em cloud no Bloco 2.

A solução permanece portável para serviços como:

- Azure Container Apps;
- AWS App Runner;
- AWS ECS/Fargate.

## Arquitetura alvo

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

GitHub Actions será responsável por:

```text
lint
testes
build
```

## Evolução planejada

### Bloco 3

- GitHub Actions;
- Airflow;
- DAG de treinamento;
- integração do treinamento existente com orquestração;
- automação de lint, testes e build.

### Bloco 4

- Prometheus;
- Grafana;
- `/metrics`;
- observabilidade;
- métricas de aplicação e inferência.

### Bloco 5

- ONNX;
- otimização do modelo;
- benchmark comparativo;
- comparação justa com o baseline;
- documentação consolidada;
- vídeo STAR.
