# Arquitetura — MedTriage MLOps

## 1. Objetivo

O MedTriage MLOps é um projeto acadêmico para classificação de urgência em textos médicos usando as classes `normal`, `attention` e `urgent`.

Essas classes são uma proxy acadêmica derivada do Medical Abstracts Text Classification Corpus e não representam um protocolo clínico validado.

## 2. Estado atual ao final do Bloco 4

```text
                         Git push / Pull Request
                                  ↓
                           GitHub Actions
                       ↙                    ↘
             quality-and-build       airflow-dag-validation
                     ↓                       ↓
          Ruff / pytest / Docker       Airflow DAG checks


Medical Abstracts TC Corpus
             ↓
      data/loader.py
             ↓
    data/validation.py
             ↓
       triage_label
             ↓
   train / validation / test
             ↓
       Apache Airflow
             ↓
        validate_data
             ↓
         run_training()
             ↓
  TF-IDF + Logistic Regression
             ↓
 baseline_pipeline.joblib
        ↙              ↘
run_evaluation()    predict.py
      ↓                ↓
evaluation.json       FastAPI
      ↓             ↙    ↓     ↘
validate_artifacts /health /predict /metrics
                              ↓
                         Prometheus
                              ↓
                           Grafana

baseline_pipeline.joblib
          ↓
benchmarking/latency.py
          ↓
baseline_latency.json
```

O Bloco 4 adiciona observabilidade HTTP sem alterar os contratos funcionais de `/health` e `/predict`.

## 3. Arquitetura alvo

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
Docker / Docker Compose
  ↓
/predict | /health | /metrics
                        ↓
                    Prometheus
                        ↓
                     Grafana
```

GitHub Actions atua transversalmente sobre qualidade, testes, Docker e validação da DAG.

## 4. Camada de dados

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

Local:

```text
data/raw/
```

Os dados não são versionados.

Contrato original:

```text
medical_abstract
condition_label
```

Proxy acadêmica:

```text
1 -> urgent
2 -> attention
3 -> attention
4 -> urgent
5 -> normal
```

## 5. Carregamento e preparação

Arquivo:

```text
src/medtriage/data/loader.py
```

Funções principais:

```text
load_dataset()
add_triage_labels()
split_training_data()
```

Responsabilidades:

- carregar CSV;
- chamar validação;
- preservar colunas originais;
- normalizar texto;
- criar `triage_label`;
- criar split reproduzível.

## 6. Validação

Arquivo:

```text
src/medtriage/data/validation.py
```

Valida:

- dataset não vazio;
- colunas obrigatórias;
- nulls;
- abstracts vazios;
- labels válidos.

## 7. Split

```text
Treino oficial: 11.550
Teste oficial:   2.888
```

Treino oficial:

```text
80% treino
20% validação
```

Reprodutibilidade:

```text
random_state = 837
```

## 8. Configuração centralizada

Arquivo:

```text
src/medtriage/config.py
```

Centraliza:

- seed;
- paths;
- colunas;
- nomes dos artefatos;
- parâmetros do baseline;
- parâmetros do benchmark.

## 9. Modelo baseline

```text
TfidfVectorizer
+
LogisticRegression
```

Persistidos em um único `sklearn.pipeline.Pipeline`.

Parâmetros:

```text
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
lowercase = True
max_iter = 1000
random_state = 837
```

## 10. Treinamento

Arquivo:

```text
src/medtriage/modeling/train.py
```

Fluxo:

```text
medical_tc_train.csv
        ↓
load_dataset()
        ↓
add_triage_labels()
        ↓
split_training_data()
        ↓
train_model()
        ↓
persist_model()
        ↓
baseline_pipeline.joblib
```

Função reutilizada pelo Airflow:

```text
run_training()
```

## 11. Avaliação

Arquivo:

```text
src/medtriage/modeling/evaluate.py
```

Fluxo:

```text
baseline_pipeline.joblib
        ↓
validation + test
        ↓
evaluate_model()
        ↓
calculate_metrics()
        ↓
evaluation.json
```

Função reutilizada pelo Airflow:

```text
run_evaluation()
```

## 12. Inferência

Arquivo:

```text
src/medtriage/modeling/predict.py
```

Responsabilidades:

- carregar modelo;
- inferir classe;
- retornar probabilidades;
- medir latência de inferência.

O modelo é carregado uma única vez no lifespan da FastAPI.

## 13. FastAPI

Arquivo:

```text
src/medtriage/api/app.py
```

Endpoints:

```text
GET  /health
POST /predict
GET  /metrics
```

Contratos funcionais existentes:

```text
/health  -> status da aplicação
/predict -> classificação, probabilidades e inference_time_ms
```

O endpoint `/metrics` pertence à camada de observabilidade.

## 14. Instrumentação Prometheus

Arquivo:

```text
src/medtriage/api/metrics.py
```

Estratégia escolhida:

```text
middleware próprio FastAPI
+
prometheus_client
```

Fluxo:

```text
request
   ↓
middleware
   ↓
endpoint
   ↓
response
   ↓
middleware registra:
- request count
- duração HTTP
- erros
```

Motivos da escolha:

- instrumentação centralizada;
- evita duplicação entre endpoints;
- baixo acoplamento;
- sem biblioteca automática adicional;
- controle explícito das labels;
- fácil preservação durante evolução futura do modelo.

## 15. Métricas

### medtriage_http_requests_total

Tipo:

```text
Counter
```

Labels:

```text
method
route
status_code
```

Significado:

```text
quantidade de requisições HTTP processadas pela API
```

### medtriage_http_request_duration_seconds

Tipo:

```text
Histogram
```

Labels:

```text
method
route
```

Significado:

```text
duração HTTP end-to-end da requisição
```

### medtriage_http_errors_total

Tipo:

```text
Counter
```

Labels:

```text
method
route
status_code
```

Significado:

```text
quantidade de respostas HTTP classificadas como erro
```

Regra:

```text
status_code >= 400
```

## 16. Estratégia de cardinalidade

Não são usados como labels:

- conteúdo do request;
- texto médico;
- IDs livres;
- mensagens arbitrárias de erro;
- qualquer dado sensível.

Rotas reconhecidas usam o path normalizado do FastAPI.

Rotas inexistentes usam:

```text
route="__unmatched__"
```

Isso evita criar uma nova série Prometheus para cada path arbitrário.

## 17. Rotas excluídas

Não são instrumentadas:

```text
/metrics
/docs
/openapi.json
```

Motivos:

- `/metrics` é consultada automaticamente pelo Prometheus;
- `/docs` e `/openapi.json` são rotas de suporte/documentação;
- a exclusão reduz ruído;
- evita o Prometheus alterar as próprias métricas a cada scrape.

## 18. Latência

Há três conceitos distintos no projeto.

### Inferência

```text
inference_time_ms
```

Medida pelo serviço de predição.

### HTTP end-to-end

```text
medtriage_http_request_duration_seconds
```

Medida pelo middleware.

### Benchmark do modelo

```text
artifacts/benchmarks/baseline_latency.json
```

Gerado por:

```text
src/medtriage/benchmarking/latency.py
```

O benchmark será usado no Bloco 5 para comparação com a versão otimizada.

## 19. Docker

O Dockerfile:

- usa `python:3.12.2-slim`;
- instala dependências;
- copia a aplicação;
- copia o modelo;
- usa usuário não-root;
- inicia Uvicorn.

Contrato de runtime:

```text
artifacts/models/baseline_pipeline.joblib
```

O modelo real deve existir localmente antes do build.

## 20. Artefato temporário para CI

Problema:

O modelo real é ignorado pelo Git, mas o Dockerfile precisa dele.

Solução:

```text
src/medtriage/ci/prepare_model.py
```

Esse módulo:

- cria dados sintéticos mínimos;
- reutiliza treino;
- reutiliza persistência;
- produz o artefato esperado pelo Dockerfile.

O artefato é temporário e exclusivo do CI.

Ele não deve substituir o modelo real na execução local da stack.

## 21. Docker Compose

Arquivo:

```text
docker-compose.yml
```

Serviços:

```text
api
prometheus
grafana
```

Portas:

```text
api         8000
prometheus  9090
grafana     3000
```

Fluxo de rede:

```text
grafana
   ↓
http://prometheus:9090
   ↓
Prometheus
   ↓
http://api:8000/metrics
   ↓
FastAPI
```

O projeto usa a rede default criada automaticamente pelo Docker Compose.

Não há necessidade atual de redes adicionais.

## 22. Prometheus

Imagem:

```text
prom/prometheus:v3.5.0
```

Configuração:

```text
monitoring/prometheus/prometheus.yml
```

Configuração efetiva:

```text
scrape_interval = 5s
job_name        = medtriage-api
target          = api:8000
metrics_path    = /metrics
```

Estado validado no Bloco 4:

```text
medtriage-api -> UP
```

O hostname `api` vem do nome do serviço no Docker Compose.

## 23. Grafana

Imagem:

```text
grafana/grafana:11.6.0
```

URL:

```text
http://localhost:3000
```

Credenciais locais:

```text
admin / admin
```

Essas credenciais existem apenas para demonstração local.

## 24. Provisioning do Grafana

Datasource:

```text
monitoring/grafana/provisioning/datasources/datasource.yml
```

Configuração:

```text
name       = Prometheus
uid        = prometheus
type       = prometheus
url        = http://prometheus:9090
isDefault  = true
editable   = false
```

Dashboard provider:

```text
monitoring/grafana/provisioning/dashboards/dashboards.yml
```

Provider:

```text
name   = MedTriage
folder = MedTriage
path   = /var/lib/grafana/dashboards
```

O provisioning garante:

```text
docker compose up
       ↓
Grafana inicia
       ↓
Prometheus já existe como datasource
       ↓
dashboard MedTriage já existe
```

Não é necessária configuração manual pela interface.

## 25. Dashboard

Arquivo versionado:

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

Refresh:

```text
5s
```

Janela padrão:

```text
últimos 15 minutos
```

### Painel 1 — Total de Requisições

Tipo:

```text
Stat
```

Query:

```promql
sum(medtriage_http_requests_total)
```

### Painel 2 — Latência HTTP p95

Tipo:

```text
Time series
```

Query:

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(medtriage_http_request_duration_seconds_bucket[5m])
  )
)
```

### Painel 3 — Taxa de Erro

Tipo:

```text
Stat
```

Query:

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

A divisão usa `clamp_min` para evitar divisão por zero em períodos sem tráfego.

## 26. Geração de tráfego

Entry point:

```text
scripts/generate_requests.py
```

Implementação:

```text
src/medtriage/monitoring/traffic.py
```

A separação mantém:

```text
scripts/
   ↓
interface de execução

src/medtriage/
   ↓
lógica reutilizável e testável
```

Parâmetros:

```text
--base-url
--requests
--delay
--invalid-ratio
```

Defaults:

```text
base_url      = http://127.0.0.1:8000
requests      = 100
delay         = 0.05
invalid_ratio = 0.1
```

Distribuição conceitual:

```text
request
  ├── /predict inválido
  ├── /health
  └── /predict válido
```

O objetivo não é executar load testing de produção.

O objetivo é gerar tráfego suficiente para:

- popular as métricas;
- demonstrar contagem de requests;
- demonstrar latência;
- demonstrar taxa de erro;
- tornar o dashboard observável durante a apresentação.

## 27. Testes da observabilidade

Integração:

```text
tests/integration/test_metrics.py
```

Valida:

- `/metrics` retorna 200;
- content type Prometheus;
- contador de requests;
- observação do histograma;
- contador de erros;
- exclusão de `/metrics`;
- exclusão de `/docs`;
- exclusão de `/openapi.json`;
- label limitada para rota desconhecida.

Unitários:

```text
tests/unit/test_traffic_generator.py
```

Valida:

- classificação de sucesso;
- classificação de erro 4xx;
- classificação de erro 5xx;
- argumentos válidos;
- rejeição de argumentos inválidos.

Ao final do Bloco 4:

```text
55 passed
```

Ruff:

```text
check -> OK
format --check -> OK
```

## 28. GitHub Actions

Arquivo:

```text
.github/workflows/ci.yml
```

O workflow continua com dois jobs:

```text
quality-and-build
airflow-dag-validation
```

Os novos testes de observabilidade entram automaticamente em:

```text
poetry run pytest
```

O Bloco 4 não adiciona Compose ao CI porque isso não é necessário para cumprir o escopo.

## 29. Airflow

Runtime separado do Poetry principal.

DAG:

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

Princípio preservado:

```text
Airflow sabe QUANDO executar.
MedTriage sabe COMO executar.
```

A observabilidade da API não altera a arquitetura do pipeline de treinamento.

## 30. Fluxo operacional completo

```text
1. Treinar modelo
   ↓
baseline_pipeline.joblib

2. docker compose up --build
   ↓
API sobe
   ↓
Prometheus sobe
   ↓
Grafana sobe

3. Prometheus coleta /metrics
   ↓
target medtriage-api fica UP

4. Grafana consulta Prometheus
   ↓
dashboard provisionado exibe dados

5. generate_requests.py
   ↓
/health + /predict válido + /predict inválido
   ↓
métricas mudam
   ↓
dashboard reage
```

## 31. Decisões arquiteturais do Bloco 4

### Middleware próprio

Escolhido em vez de instrumentação manual por endpoint ou biblioteca automática externa.

Motivo:

- centralização;
- simplicidade;
- baixo acoplamento;
- controle de labels.

### Erro = status >= 400

Motivo:

- regra simples;
- captura 4xx e 5xx;
- permite demonstrar erros controlados.

### Exclusão de rotas de infraestrutura

Excluídas:

```text
/metrics
/docs
/openapi.json
```

Motivo:

- reduzir ruído;
- evitar auto-instrumentação do scrape;
- aproximar as métricas do uso funcional.

### Compose simples

Escolha:

- rede default;
- sem healthchecks adicionais;
- sem restart policies;
- sem persistência adicional;
- sem serviços extras.

Motivo:

- evitar overengineering;
- manter stack reprodutível e adequada ao Tech Challenge.

### Provisioning automático

Datasource e dashboard são versionados.

Motivo:

- reprodutibilidade;
- facilidade de avaliação;
- nenhuma configuração manual necessária.

## 32. Limitações deliberadas

Fora do Bloco 4:

- Alertmanager;
- Loki;
- OpenTelemetry;
- Elasticsearch;
- Kubernetes;
- drift monitoring;
- autenticação de produção;
- persistência dedicada;
- monitoramento avançado de ML.

Esses itens não são necessários para os requisitos atuais.

## 33. Continuidade para o Bloco 5

O Bloco 5 deverá implementar:

- ONNX;
- otimização;
- benchmark comparativo;
- documentação final;
- vídeo STAR.

Regras de continuidade:

1. preservar `/health`;
2. preservar `/predict`;
3. preservar `/metrics`;
4. preservar os nomes das métricas;
5. preservar as labels;
6. evitar duplicar instrumentação;
7. preservar o Docker Compose;
8. manter Prometheus e Grafana funcionais;
9. usar o benchmark baseline existente como referência;
10. atualizar a inferência sem quebrar o contrato de resposta.

Arquivos que provavelmente serão impactados:

```text
src/medtriage/modeling/predict.py
src/medtriage/config.py
src/medtriage/benchmarking/
Dockerfile
pyproject.toml
poetry.lock
README.md
docs/architecture.md
```

Arquivos de observabilidade que idealmente devem permanecer estáveis:

```text
src/medtriage/api/metrics.py
monitoring/prometheus/prometheus.yml
monitoring/grafana/provisioning/
```

O dashboard só deverá ser alterado se houver uma necessidade real de visualizar novas métricas.
