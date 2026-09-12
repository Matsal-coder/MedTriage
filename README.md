# MedTriage MLOps

Projeto desenvolvido para o Tech Challenge da Fase 3 da Pós-Tech FIAP.

O objetivo é construir uma solução de MLOps para classificação acadêmica de urgência em textos médicos, incluindo treinamento reproduzível, API de inferência, containerização, CI/CD, orquestração, observabilidade e otimização/avaliação de latência.

> IMPORTANTE
>
> O mapeamento para as classes `normal`, `attention` e `urgent` é uma simplificação acadêmica criada para demonstrar a arquitetura MLOps. Ele NÃO representa uma regra clínica validada e NÃO deve ser interpretado como um sistema médico de triagem real.

## Status do projeto

O projeto está na etapa final do BLOCO 5 — otimização com ONNX, benchmark comparativo e documentação final.

Principais componentes implementados:

- estrutura Python em layout `src/`;
- gerenciamento de dependências com Poetry;
- Python 3.12.2;
- testes com pytest;
- lint e formatação com Ruff;
- configuração centralizada;
- logging centralizado;
- dataset Medical Abstracts Text Classification Corpus;
- validação e preparação dos dados;
- split reproduzível treino/validação;
- baseline TF-IDF + Logistic Regression;
- persistência do baseline com Joblib;
- avaliação em validação e teste;
- FastAPI;
- endpoints `GET /health`, `POST /predict` e `GET /metrics`;
- Dockerfile com execução non-root;
- Docker Compose;
- GitHub Actions;
- Apache Airflow 3.3.1;
- pipeline de treino e avaliação;
- Prometheus;
- Grafana;
- dashboard provisionado automaticamente;
- gerador de tráfego;
- conversão ONNX;
- runtime ONNX CPU;
- testes de equivalência sklearn vs ONNX;
- benchmark comparativo de latência;
- cálculo de speedup;
- 84 testes automatizados.

MLflow foi deliberadamente excluído da arquitetura deste projeto.

## Stack

### Aplicação e ML

- Python 3.12.2
- Poetry 2.4.3
- pandas
- scikit-learn
- Joblib
- NumPy
- ONNX
- skl2onnx
- ONNX Runtime CPU

### API e qualidade

- FastAPI
- Uvicorn
- pytest
- Ruff

### Infraestrutura e MLOps

- Docker
- Docker Compose
- GitHub Actions
- Apache Airflow 3.3.1
- WSL2 / Ubuntu 24.04 LTS para execução local do Airflow
- prometheus-client
- Prometheus 3.5.0
- Grafana 11.6.0

## Estrutura do projeto

```text
medtriage-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── airflow/
│   ├── README.md
│   └── requirements-airflow.txt
├── artifacts/
│   ├── benchmarks/
│   │   ├── baseline_latency.json
│   │   ├── optimized_latency.json
│   │   └── latency_comparison.json
│   └── models/
│       ├── baseline_pipeline.joblib
│       ├── optimized_model.onnx
│       └── evaluation.json
├── dags/
│   └── training_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── architecture.md
│   ├── baseline-results.md
│   └── latency-comparison.md
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       │   └── medtriage-dashboard.json
│       └── provisioning/
│           ├── dashboards/
│           │   └── dashboards.yml
│           └── datasources/
│               └── datasource.yml
├── scripts/
│   └── generate_requests.py
├── src/
│   └── medtriage/
│       ├── api/
│       │   ├── app.py
│       │   ├── metrics.py
│       │   └── schemas.py
│       ├── benchmarking/
│       │   ├── latency.py
│       │   └── comparison.py
│       ├── ci/
│       │   └── prepare_model.py
│       ├── data/
│       ├── modeling/
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── predict.py
│       │   ├── onnx_export.py
│       │   └── onnx_predict.py
│       ├── monitoring/
│       │   └── traffic.py
│       ├── config.py
│       └── logging.py
├── tests/
│   ├── integration/
│   └── unit/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── poetry.lock
└── README.md
```

Os datasets locais e os artefatos intermediários permanecem ignorados pelo Git.

Como evidência da entrega final, são versionados explicitamente:

```text
artifacts/models/optimized_model.onnx
artifacts/benchmarks/baseline_latency.json
artifacts/benchmarks/optimized_latency.json
artifacts/benchmarks/latency_comparison.json
```

O baseline `baseline_pipeline.joblib`, `evaluation.json` e demais artefatos gerados localmente permanecem ignorados.

## Instalação

```bash
poetry install
```

O pacote `medtriage` é instalado a partir de `src/medtriage/`.

## Qualidade e testes

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
```

Estado atual:

```text
84 passed
```

Os warnings remanescentes são depreciações provenientes de dependências Starlette/AnyIO e não representam falhas funcionais do projeto.

## Dataset

Dataset utilizado:

```text
Medical Abstracts Text Classification Corpus
```

Arquivos:

```text
medical_tc_train.csv
medical_tc_test.csv
medical_tc_labels.csv
```

Local:

```text
data/raw/
```

Os dados não são versionados no Git.

Colunas principais:

```text
medical_abstract
condition_label
triage_label
```

Mapeamento acadêmico utilizado:

```text
1 -> urgent
2 -> attention
3 -> attention
4 -> urgent
5 -> normal
```

Essa transformação existe somente para o objetivo acadêmico do Tech Challenge.

## Pipeline de dados

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

O conjunto de treino oficial possui 11.550 amostras e é dividido em 80% treino e 20% validação, com estratificação e `RANDOM_SEED = 837`.

O conjunto oficial de teste possui 2.888 amostras.

## Modelo baseline

A pipeline original utiliza:

```text
TfidfVectorizer
        +
LogisticRegression
```

Parâmetros principais:

```text
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
lowercase = True
max_iter = 1000
random_state = 837
```

Treinamento:

```bash
poetry run python -m medtriage.modeling.train
```

Artefato:

```text
artifacts/models/baseline_pipeline.joblib
```

## Avaliação

Execução:

```bash
poetry run python -m medtriage.modeling.evaluate
```

Artefato:

```text
artifacts/models/evaluation.json
```

Resultados no conjunto oficial de teste:

```text
samples           2888
accuracy          0.5765
precision_macro   0.5524
recall_macro      0.5447
f1_macro          0.5457
f1_weighted       0.5684
urgent_recall     0.7675
```

Esses resultados não devem ser interpretados como validação clínica.

## API local

Inicialização:

```bash
poetry run uvicorn medtriage.api.app:app --host 127.0.0.1 --port 8000
```

Endpoints:

```text
GET  /health
POST /predict
GET  /metrics
```

### GET /health

Resposta:

```json
{"status":"ok"}
```

### POST /predict

Request:

```json
{
  "text": "Patient with severe chest pain and shortness of breath."
}
```

Contrato de resposta:

```json
{
  "prediction": "attention",
  "probabilities": {
    "attention": 0.44,
    "normal": 0.35,
    "urgent": 0.21
  },
  "inference_time_ms": 2.8
}
```

`inference_time_ms` mede a inferência do backend de modelo e não a duração HTTP completa.

### GET /metrics

Expõe métricas Prometheus.

Principais métricas:

```text
medtriage_http_requests_total
medtriage_http_request_duration_seconds
medtriage_http_errors_total
```

## Observabilidade

A instrumentação HTTP utiliza middleware FastAPI com `prometheus_client`.

Decisões principais:

- `Counter` para número de requisições;
- `Histogram` para duração HTTP;
- `Counter` para erros;
- erro definido como `status_code >= 400`;
- labels de baixa cardinalidade;
- rota normalizada;
- rotas não reconhecidas usam `route="__unmatched__"`.

Labels utilizadas:

```text
method
route
status_code
```

Não são usados como labels:

- texto médico;
- conteúdo livre do request;
- identificadores livres;
- mensagens arbitrárias de erro;
- dados sensíveis.

Rotas excluídas:

```text
/metrics
/docs
/openapi.json
```

## Latência HTTP vs latência de inferência

São métricas diferentes.

`inference_time_ms`:

```text
tempo gasto no caminho de inferência do modelo
```

`medtriage_http_request_duration_seconds`:

```text
tempo HTTP end-to-end
```

O benchmark de modelo é mantido separadamente em `artifacts/benchmarks/`.

## Docker

Build:

```bash
docker build -t medtriage-mlops .
```

Execução:

```bash
docker run --rm -p 8000:8000 medtriage-mlops
```

O container preserva execução non-root e expõe a API na porta 8000.

## Docker Compose

Subida do ambiente:

```bash
docker compose up --build
```

Serviços:

```text
API         http://localhost:8000
Prometheus  http://localhost:9090
Grafana     http://localhost:3000
```

O Prometheus coleta:

```text
api:8000/metrics
```

O Grafana utiliza o Prometheus como datasource interno em:

```text
http://prometheus:9090
```

## GitHub Actions

Workflow:

```text
.github/workflows/ci.yml
```

Jobs:

```text
quality-and-build
airflow-dag-validation
```

`quality-and-build` executa:

```text
checkout
Python 3.12.2
Poetry 2.4.3
poetry install
Ruff lint
Ruff format check
pytest
modelo sintético temporário
Docker build
```

O modelo sintético de CI permite validar o Docker em runner limpo sem depender do dataset real.

`airflow-dag-validation` instala o Airflow em ambiente isolado e valida:

```text
import da DAG
medtriage_training_pipeline
validate_data
train_model
evaluate_model
validate_artifacts
```

## Airflow

DAG:

```text
medtriage_training_pipeline
```

Configuração:

```text
schedule=None
catchup=False
```

Fluxo:

```text
validate_data
    ↓
train_model
    ↓
evaluate_model
    ↓
validate_artifacts
```

Princípio arquitetural:

```text
Airflow sabe QUANDO executar.
MedTriage sabe COMO executar.
```

A lógica de ML permanece em `src/medtriage/` e a DAG apenas orquestra funções existentes.

## Prometheus

Imagem:

```text
prom/prometheus:v3.5.0
```

Configuração:

```text
scrape_interval = 5s
job_name = medtriage-api
target = api:8000
metrics_path = /metrics
```

## Grafana

Imagem:

```text
grafana/grafana:11.6.0
```

Dashboard:

```text
MedTriage API Monitoring
```

UID:

```text
medtriage-api-monitoring
```

Painéis:

1. Total de Requisições
2. Latência HTTP p95
3. Taxa de Erro

Datasource e dashboard são provisionados automaticamente.

## Gerador de tráfego

Script:

```text
scripts/generate_requests.py
```

Implementação:

```text
src/medtriage/monitoring/traffic.py
```

É utilizado para gerar tráfego controlado e demonstrar as métricas no Prometheus/Grafana.

# Otimização com ONNX

## Estratégia inicialmente avaliada

A primeira tentativa converteu toda a pipeline para ONNX:

```text
texto
 ↓
TfidfVectorizer ONNX
 ↓
LogisticRegression ONNX
```

A conversão estrutural foi bem-sucedida:

- artefato ONNX criado;
- `onnx.checker` validou o modelo;
- ONNX Runtime carregou o modelo;
- classes previstas permaneceram coerentes.

Entretanto, os testes de equivalência mostraram divergências relevantes nas probabilidades.

A diferença absoluta máxima observada nos testes sintéticos chegou a aproximadamente:

```text
0.0689
```

Por esse motivo, a pipeline ONNX completa foi rejeitada. A tolerância dos testes não foi artificialmente aumentada.

## Arquitetura otimizada final

A arquitetura escolhida preserva o `TfidfVectorizer` original e converte somente a `LogisticRegression`:

```text
texto
 ↓
TfidfVectorizer sklearn
 ↓
matriz CSR
 ↓
float32 / dense tensor
 ↓
LogisticRegression ONNX Runtime
 ↓
prediction + probabilities
```

Motivo:

- preservar o preprocessing exato do baseline;
- manter as classes previstas;
- preservar ordem de classes;
- preservar probabilidades dentro de tolerância pequena;
- isolar a mudança de runtime no classificador.

No modelo real, o TF-IDF possui:

```text
186.957 features
```

O classificador ONNX recebe:

```text
features: [None, 186957] tensor(float)
```

e retorna:

```text
label         [None]    tensor(string)
probabilities [None, 3] tensor(float)
```

Backend utilizado:

```text
CPUExecutionProvider
```

## Equivalência sklearn vs ONNX

A equivalência foi validada por testes automatizados.

Critérios:

- mesmas classes;
- mesma ordem das classes;
- mesma classe prevista;
- mesmas labels de probabilidades;
- probabilidades dentro de tolerância absoluta de `1e-5`.

Os testes de equivalência passaram integralmente.

## Benchmark de latência

Metodologia oficial:

```text
escopo: model_inference
inputs: 5 textos fixos
warm-up: 20 execuções
medições: 500
mesma máquina
mesmo preprocessing
modelo carregado previamente
```

Para reduzir efeitos de ambiente, a comparação final executa baseline e backend ONNX de forma pareada no mesmo ambiente.

### Resultado pareado final

| Métrica | Baseline sklearn | Backend ONNX |
|---|---:|---:|
| mean | 2.9587 ms | 2.8603 ms |
| p50 | 2.7022 ms | 2.3932 ms |
| p95 | 3.8490 ms | 5.1237 ms |
| min | 2.3885 ms | 1.3688 ms |
| max | 6.7184 ms | 19.6125 ms |
| throughput | 337.99 req/s | 349.62 req/s |

Speedups:

```text
mean_speedup = 1.0344x
p50_speedup  = 1.1291x
p95_speedup  = 0.7512x
```

Fórmula:

```text
speedup = baseline_latency / optimized_latency
```

Interpretação:

- `> 1.0x`: otimizado mais rápido;
- `= 1.0x`: equivalência;
- `< 1.0x`: otimizado mais lento.

## Resultado da otimização

O ONNX foi aplicado e validado funcionalmente. Na execução pareada final, apresentou pequena melhora na média e no p50, mas piorou de forma relevante o p95 e mostrou maior variabilidade de latência.

A principal explicação arquitetural é que a regressão logística original já possui baixo custo computacional, enquanto o backend ONNX precisa converter a saída esparsa do TF-IDF para um tensor denso `float32` de alta dimensionalidade.

Portanto, neste cenário específico, o ganho potencial do runtime ONNX compete diretamente com o custo adicional de preparação do tensor.

Como o resultado não demonstrou ganho consistente em toda a distribuição de latência, a API final permanece utilizando o backend sklearn. A implementação ONNX é mantida como alternativa validada, entregue e benchmarkada.

Os resultados foram preservados e documentados sem manipulação do benchmark.

## Artefatos de benchmark

```text
artifacts/benchmarks/baseline_latency.json
artifacts/benchmarks/optimized_latency.json
artifacts/benchmarks/latency_comparison.json
```

Os artefatos intermediários permanecem ignorados pelo Git. Como evidência da entrega final, são versionados explicitamente:

```text
artifacts/models/optimized_model.onnx
artifacts/benchmarks/baseline_latency.json
artifacts/benchmarks/optimized_latency.json
artifacts/benchmarks/latency_comparison.json
```

## Reproduzindo a conversão ONNX

```bash
poetry run python -m medtriage.modeling.onnx_export
```

Artefato produzido:

```text
artifacts/models/optimized_model.onnx
```

## Reproduzindo o benchmark comparativo

```bash
poetry run python -m medtriage.benchmarking.comparison
```

O comando executa baseline e ONNX no mesmo ambiente e gera os artefatos de comparação.

## Limitações

- o problema de triagem é uma proxy acadêmica, não um protocolo clínico;
- as classes derivam de um mapeamento simplificado das labels do dataset;
- métricas de classificação não representam validação clínica;
- ONNX não apresentou ganho consistente de latência no cenário final;
- a representação TF-IDF possui alta dimensionalidade;
- o backend ONNX híbrido exige conversão de matriz esparsa para tensor denso;
- benchmark local depende do hardware e da carga do sistema;
- resultados de latência não devem ser generalizados para outras máquinas;
- o dataset, o baseline sklearn e artefatos intermediários não são versionados; o modelo ONNX final e os JSONs de benchmark são versionados como evidência da entrega;
- o Airflow local é executado em ambiente isolado do Poetry principal;
- o projeto utiliza CPU e não depende de GPU.

## Decisão de cloud

O objetivo do projeto é demonstrar arquitetura MLOps reproduzível e containerizada.

A aplicação foi estruturada de forma portável via Docker, podendo ser adaptada a um ambiente cloud, mas a entrega atual prioriza execução local/reproduzível.

## Vídeo STAR

O vídeo final deve apresentar:

### Situation
- problema de classificação acadêmica de urgência em textos médicos;
- necessidade de servir o modelo com baixa latência e observabilidade.

### Task
- treinar e servir o modelo;
- automatizar validações;
- orquestrar pipeline;
- monitorar a API;
- avaliar otimização de performance.

### Action
- TF-IDF + Logistic Regression;
- FastAPI;
- Docker;
- GitHub Actions;
- Airflow;
- Prometheus;
- Grafana;
- ONNX Runtime;
- benchmark comparativo.

### Result
- pipeline reproduzível;
- API funcional;
- CI verde;
- observabilidade completa;
- equivalência ONNX validada;
- benchmark pareado;
- ganho em média/p50, regressão no p95 e ausência de ganho consistente documentados com transparência.

Link do vídeo:

```text
INSERIR LINK FINAL DO VÍDEO
```

## Execução resumida

Instalar:

```bash
poetry install
```

Treinar:

```bash
poetry run python -m medtriage.modeling.train
```

Avaliar:

```bash
poetry run python -m medtriage.modeling.evaluate
```

Converter para ONNX:

```bash
poetry run python -m medtriage.modeling.onnx_export
```

Executar benchmark:

```bash
poetry run python -m medtriage.benchmarking.comparison
```

Subir API:

```bash
poetry run uvicorn medtriage.api.app:app --host 127.0.0.1 --port 8000
```

Subir stack de observabilidade:

```bash
docker compose up --build
```

Validar qualidade:

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
```
