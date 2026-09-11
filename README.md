# MedTriage MLOps

Projeto desenvolvido para o Tech Challenge da Fase 3 da Pós-Tech FIAP.

O objetivo geral é construir uma solução de MLOps para classificação acadêmica de urgência em textos médicos, incluindo API de inferência, containerização, automação de pipelines, observabilidade e otimização de latência.

IMPORTANTE:
O mapeamento para as classes `normal`, `attention` e `urgent` é uma simplificação acadêmica criada para demonstrar a arquitetura MLOps. Ele NÃO representa uma regra clínica validada e NÃO deve ser interpretado como sistema médico de triagem real.

## Status do projeto

O projeto está no final do BLOCO 4 — Prometheus, Grafana e observabilidade.

Neste estágio já estão implementados:

- estrutura Python em layout `src/`;
- gerenciamento de dependências com Poetry;
- lint e formatação com Ruff;
- testes com pytest;
- configuração centralizada;
- logging básico;
- aplicação FastAPI;
- endpoints `GET /health`, `POST /predict` e `GET /metrics`;
- Dockerfile funcional com usuário não-root;
- Medical Abstracts TC Corpus;
- validação e preparação dos dados;
- split reproduzível treino/validação;
- baseline TF-IDF + Logistic Regression;
- persistência com Joblib;
- avaliação em validação e teste;
- benchmark baseline de latência;
- GitHub Actions;
- validação automática de lint, formatação, testes e Docker build;
- artefato temporário de modelo para CI;
- Apache Airflow 3.3.1 em WSL/Linux;
- DAG de treino e avaliação;
- testes estruturais da DAG;
- validação real da DAG no GitHub Actions;
- execução end-to-end da DAG localmente;
- instrumentação HTTP com `prometheus-client`;
- métricas de requisições, latência e erros;
- Prometheus em Docker Compose;
- Grafana em Docker Compose;
- datasource Prometheus provisionado automaticamente;
- dashboard Grafana versionado e provisionado automaticamente;
- gerador de tráfego para demonstração;
- testes unitários e de integração da camada de observabilidade.

Ainda não fazem parte do estado atual:

- ONNX;
- quantização;
- pruning;
- benchmark comparativo baseline vs modelo otimizado;
- vídeo STAR final.

MLflow não faz parte da arquitetura deste projeto.

## Stack

### Implementada

- Python 3.12.2
- Poetry 2.4.3
- FastAPI
- Uvicorn
- pandas
- scikit-learn
- Joblib
- pytest
- Ruff
- Docker
- Docker Compose
- GitHub Actions
- Apache Airflow 3.3.1
- WSL2 / Ubuntu 24.04 LTS para Airflow local
- prometheus-client
- Prometheus 3.5.0
- Grafana 11.6.0

### Planejada para o Bloco 5

- ONNX
- otimização de modelo
- benchmark comparativo baseline vs modelo otimizado
- documentação final
- vídeo STAR

## Estrutura atual

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
│   │   └── baseline_latency.json
│   └── models/
│       ├── baseline_pipeline.joblib
│       └── evaluation.json
├── dags/
│   └── training_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── architecture.md
│   └── baseline-results.md
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
│       ├── ci/
│       ├── data/
│       ├── modeling/
│       ├── monitoring/
│       │   └── traffic.py
│       ├── config.py
│       └── logging.py
├── tests/
│   ├── integration/
│   │   └── test_metrics.py
│   └── unit/
│       └── test_traffic_generator.py
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Instalação principal

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

Ao final do Bloco 4, a suíte possui 55 testes.

## Dataset

Dataset: Medical Abstracts Text Classification Corpus.

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

Mapeamento acadêmico:

```text
1 -> urgent
2 -> attention
3 -> attention
4 -> urgent
5 -> normal
```

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

O treino oficial contém 11.550 amostras e é dividido em 80% treino e 20% validação, com `RANDOM_SEED = 837` e estratificação. O teste oficial possui 2.888 amostras.

## Modelo baseline

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

Avaliação:

```bash
poetry run python -m medtriage.modeling.evaluate
```

Artefato:

```text
artifacts/models/evaluation.json
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

Resposta esperada:

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

Response conceitual:

```json
{
  "prediction": "attention",
  "probabilities": {
    "attention": 0.44,
    "normal": 0.35,
    "urgent": 0.21
  },
  "inference_time_ms": 19.5
}
```

`inference_time_ms` mede somente a inferência do modelo.

### GET /metrics

Expõe métricas no formato Prometheus.

As principais métricas customizadas são:

```text
medtriage_http_requests_total
medtriage_http_request_duration_seconds
medtriage_http_errors_total
```

## Estratégia de instrumentação

A instrumentação HTTP é centralizada em middleware próprio com `prometheus_client`.

Arquivo:

```text
src/medtriage/api/metrics.py
```

Decisões:

- `Counter` para número de requisições;
- `Histogram` para duração HTTP;
- `Counter` para erros;
- erro definido como `status_code >= 400`;
- labels de baixa cardinalidade;
- rota normalizada em vez do path bruto;
- rotas desconhecidas usam `route="__unmatched__"`.

Labels utilizadas:

```text
method
route
status_code
```

Não são usados como labels:

- conteúdo do request;
- texto médico;
- IDs livres;
- mensagens arbitrárias de erro;
- dados sensíveis.

## Rotas excluídas da instrumentação

As seguintes rotas não entram nas métricas HTTP da aplicação:

```text
/metrics
/docs
/openapi.json
```

Motivo:

- `/metrics` é consultada periodicamente pelo próprio Prometheus;
- `/docs` e `/openapi.json` são rotas de infraestrutura/documentação;
- excluí-las reduz ruído nas métricas de uso real da API.

## Latência HTTP vs latência de inferência

Há dois conceitos diferentes:

```text
inference_time_ms
```

Tempo gasto especificamente na inferência do modelo em `/predict`.

```text
medtriage_http_request_duration_seconds
```

Tempo HTTP end-to-end da requisição.

O benchmark puro do modelo permanece separado em:

```text
artifacts/benchmarks/baseline_latency.json
```

e será usado no Bloco 5 para comparação com a versão otimizada.

## Docker

Build manual:

```bash
docker build -t medtriage-mlops .
```

Execução:

```bash
docker run --rm -p 8000:8000 medtriage-mlops
```

O container executa como usuário não-root.

## Pré-requisito do modelo

O Dockerfile copia:

```text
artifacts/models/baseline_pipeline.joblib
```

O arquivo real do modelo não é versionado no Git.

Antes de executar:

```bash
docker compose up --build
```

o artefato deve existir localmente.

Treinamento:

```bash
poetry run python -m medtriage.modeling.train
```

Validação simples:

```bash
test -f artifacts/models/baseline_pipeline.joblib && echo "model OK"
```

O modelo sintético criado em `src/medtriage/ci/prepare_model.py` serve somente ao CI e não deve ser usado como modelo real da aplicação.

## Docker Compose — observabilidade

A stack de monitoramento possui três serviços:

```text
api
prometheus
grafana
```

Inicialização:

```bash
docker compose up --build
```

Estado:

```bash
docker compose ps
```

Encerramento:

```bash
docker compose down
```

Portas:

```text
FastAPI     http://localhost:8000
Prometheus  http://localhost:9090
Grafana     http://localhost:3000
```

Fluxo:

```text
FastAPI
  ↓
GET /metrics
  ↓
Prometheus
  ↓
Grafana
```

## Prometheus

Configuração:

```text
monitoring/prometheus/prometheus.yml
```

Parâmetros principais:

```text
scrape_interval = 5s
job_name        = medtriage-api
metrics_path    = /metrics
target          = api:8000
```

Dentro da rede Docker, `api` é resolvido pelo nome do serviço do Compose.

Para validar:

1. abra `http://localhost:9090`;
2. acesse a tela de targets;
3. confirme `medtriage-api` como `UP`.

Queries úteis:

```promql
medtriage_http_requests_total
```

```promql
medtriage_http_request_duration_seconds_count
```

```promql
medtriage_http_errors_total
```

## Grafana

Imagem:

```text
grafana/grafana:11.6.0
```

Acesso:

```text
http://localhost:3000
```

Credenciais locais de demonstração:

```text
usuário: admin
senha: admin
```

Essas credenciais são apenas locais e não representam uma configuração adequada para produção.

### Datasource

Arquivo:

```text
monitoring/grafana/provisioning/datasources/datasource.yml
```

Datasource:

```text
name: Prometheus
uid: prometheus
url: http://prometheus:9090
default: true
```

### Dashboard provisioning

Arquivo:

```text
monitoring/grafana/provisioning/dashboards/dashboards.yml
```

Pasta provisionada:

```text
/var/lib/grafana/dashboards
```

Dashboard versionado:

```text
monitoring/grafana/dashboards/medtriage-dashboard.json
```

Título:

```text
MedTriage API Monitoring
```

UID:

```text
medtriage-api-monitoring
```

O dashboard é carregado automaticamente quando o Grafana inicia.

## Dashboard de monitoramento

O dashboard possui três painéis mínimos.

### 1. Total de Requisições

```promql
sum(medtriage_http_requests_total)
```

### 2. Latência HTTP p95

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(medtriage_http_request_duration_seconds_bucket[5m])
  )
)
```

### 3. Taxa de Erro

```promql
100
*
sum(rate(medtriage_http_errors_total[5m]))
/
clamp_min(
  sum(rate(medtriage_http_requests_total[5m])),
  0.000000001
)
```

A taxa de erro considera respostas HTTP com status `>= 400`.

## Gerador de tráfego

Entry point:

```text
scripts/generate_requests.py
```

Implementação:

```text
src/medtriage/monitoring/traffic.py
```

Execução padrão:

```bash
poetry run python scripts/generate_requests.py
```

Por padrão:

```text
requests      = 100
delay         = 0.05 s
invalid_ratio = 0.1
```

Exemplo:

```bash
poetry run python scripts/generate_requests.py \
  --requests 200 \
  --delay 0.02 \
  --invalid-ratio 0.1
```

O script gera uma combinação de:

- `GET /health`;
- `POST /predict` válido;
- `POST /predict` inválido para produzir erros controlados.

Exemplo de saída:

```text
MedTriage traffic generation complete
========================================
Total requests      : 200
Successful          : 178
Client errors       : 22
Server errors       : 0
Unexpected failures : 0
```

Cenário sem erros intencionais:

```bash
poetry run python scripts/generate_requests.py \
  --requests 50 \
  --invalid-ratio 0
```

Cenário com maior taxa de erro:

```bash
poetry run python scripts/generate_requests.py \
  --requests 50 \
  --invalid-ratio 0.3
```

## Validação da stack de observabilidade

1. Garantir que o modelo existe:

```bash
test -f artifacts/models/baseline_pipeline.joblib && echo "model OK"
```

2. Validar Compose:

```bash
docker compose config
```

3. Subir stack:

```bash
docker compose up --build
```

4. Validar API:

```bash
curl http://127.0.0.1:8000/health
```

5. Validar inferência:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Patient with severe chest pain and shortness of breath."}'
```

6. Validar métricas:

```bash
curl http://127.0.0.1:8000/metrics
```

7. Confirmar Prometheus:

```text
http://localhost:9090
```

Target esperado:

```text
medtriage-api -> UP
```

8. Confirmar Grafana:

```text
http://localhost:3000
```

Dashboard esperado:

```text
MedTriage API Monitoring
```

9. Gerar tráfego:

```bash
poetry run python scripts/generate_requests.py --requests 200
```

10. Verificar os painéis.

## Estratégia de artefato para CI

O Dockerfile copia:

```text
artifacts/models/baseline_pipeline.joblib
```

Esse arquivo não é versionado. Para permitir Docker build em runner limpo, foi criado:

```text
src/medtriage/ci/prepare_model.py
```

A rotina cria um modelo temporário usando o mesmo código de treino e persistência do projeto. Ele serve somente ao CI e não substitui o modelo real.

## GitHub Actions

Workflow:

```text
.github/workflows/ci.yml
```

Triggers:

```text
push -> main
pull_request -> main
```

### Job `quality-and-build`

```text
checkout
 ↓
Python 3.12.2
 ↓
Poetry 2.4.3
 ↓
poetry install
 ↓
ruff check
 ↓
ruff format --check
 ↓
pytest
 ↓
prepare_model
 ↓
docker build
```

Os testes adicionados no Bloco 4 entram automaticamente na suíte executada por esse job.

### Job `airflow-dag-validation`

Mantém a validação independente do DAG do Airflow.

## Airflow

Runtime separado do Poetry principal.

Ambiente local:

```text
WSL2
Ubuntu 24.04
Python 3.12
Apache Airflow 3.3.1
```

DAG:

```text
dags/training_pipeline.py
```

DAG ID:

```text
medtriage_training_pipeline
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

## Limitações deliberadas

O Bloco 4 não implementa:

- Alertmanager;
- OpenTelemetry;
- Loki;
- Elasticsearch;
- Kubernetes;
- drift monitoring;
- monitoramento avançado de ML;
- autenticação de produção no Grafana;
- persistência dedicada para Prometheus/Grafana.

Esses elementos não são necessários para o escopo acadêmico atual e foram evitados para reduzir complexidade desnecessária.

## Próximo passo — Bloco 5

O Bloco 5 será responsável por:

- ONNX;
- otimização do modelo;
- benchmark comparativo;
- preservação dos contratos `/health`, `/predict` e `/metrics`;
- preservação das métricas e labels existentes;
- documentação final;
- vídeo STAR.

Ao otimizar o modelo, a camada de observabilidade criada neste bloco deve permanecer funcional sem alteração de seus contratos.
