# Arquitetura — MedTriage MLOps

## 1. Objetivo

O MedTriage MLOps é um projeto acadêmico para classificação de urgência em textos médicos usando as classes:

```text
normal
attention
urgent
```

Essas classes são uma proxy acadêmica derivada do Medical Abstracts Text Classification Corpus e não representam protocolo clínico validado.

## 2. Arquitetura final

```text
                         Git push / Pull Request
                                  ↓
                            GitHub Actions
                        ↙                    ↘
              quality-and-build       airflow-dag-validation
                      ↓                       ↓
           Ruff / pytest / Docker       Airflow DAG checks


Medical Abstracts Text Classification Corpus
                    ↓
            data/loader.py
                    ↓
          data/validation.py
                    ↓
              triage_label
                    ↓
        treino / validação / teste
                    ↓
              Apache Airflow
                    ↓
              validate_data
                    ↓
               train_model
                    ↓
          TF-IDF + LogisticRegression
                    ↓
       baseline_pipeline.joblib
             ↙              ↘
   evaluation.py         inference
        ↓                  ↙   ↘
 evaluation.json    sklearn    ONNX
                        ↓       ↓
                     FastAPI / benchmark
                          ↓
               /health /predict /metrics
                          ↓
                      Prometheus
                          ↓
                        Grafana
```

A otimização ONNX final utiliza uma arquitetura híbrida:

```text
texto
  ↓
TfidfVectorizer sklearn
  ↓
matriz CSR
  ↓
conversão float32 + dense
  ↓
LogisticRegression ONNX Runtime
  ↓
prediction + probabilities
```

## 3. Princípios arquiteturais

### Separação de responsabilidades

- `data/`: carregamento e validação;
- `modeling/`: treino, avaliação, inferência e exportação ONNX;
- `benchmarking/`: medição de latência e comparação;
- `api/`: contrato HTTP e instrumentação;
- `monitoring/`: geração de tráfego;
- `ci/`: artefato sintético para validações;
- `dags/`: apenas orquestração;
- `config.py`: centralização de paths e parâmetros.

### Airflow

Princípio:

```text
Airflow sabe QUANDO executar.
MedTriage sabe COMO executar.
```

A DAG não duplica lógica de machine learning.

### Observabilidade

Prometheus/Grafana observam a API sem alterar o contrato funcional de `/health` e `/predict`.

### Otimização

A corretude tem prioridade sobre speedup.

Uma conversão ONNX só é aceita se preservar o comportamento do baseline dentro de tolerância definida.

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

## 6. Validação de dados

Arquivo:

```text
src/medtriage/data/validation.py
```

Valida:

- dataset não vazio;
- colunas obrigatórias;
- nulls;
- abstracts vazios;
- labels válidas.

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
- paths ONNX;
- parâmetros de benchmark;
- nomes dos artefatos de comparação.

## 9. Modelo baseline

Pipeline:

```text
TfidfVectorizer
        +
LogisticRegression
```

Persistência:

```text
artifacts/models/baseline_pipeline.joblib
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

## 12. Inferência baseline

Arquivo:

```text
src/medtriage/modeling/predict.py
```

Responsabilidades:

- carregar pipeline sklearn;
- executar predição;
- obter probabilidades;
- medir latência de inferência.

O serviço é carregado uma única vez no lifespan da FastAPI.

## 13. Exportação ONNX

Arquivo:

```text
src/medtriage/modeling/onnx_export.py
```

A primeira abordagem tentou converter:

```text
TfidfVectorizer + LogisticRegression
```

inteiramente para ONNX.

Embora o artefato fosse estruturalmente válido e carregável pelo ONNX Runtime, os testes identificaram divergência probabilística de até aproximadamente:

```text
0.0689
```

A arquitetura foi rejeitada.

A versão final exporta apenas:

```text
LogisticRegression
```

Input real:

```text
features: [None, 186957] tensor(float)
```

Outputs:

```text
label         [None]    tensor(string)
probabilities [None, 3] tensor(float)
```

Provider:

```text
CPUExecutionProvider
```

## 14. Inferência ONNX final

Arquivo:

```text
src/medtriage/modeling/onnx_predict.py
```

Fluxo:

```text
texto
 ↓
TfidfVectorizer sklearn
 ↓
sparse CSR float64
 ↓
astype(float32)
 ↓
toarray()
 ↓
ONNX Runtime
 ↓
LogisticRegression ONNX
 ↓
label + probabilities
```

O runtime carrega:

```text
baseline_pipeline.joblib
```

somente para reutilizar o `TfidfVectorizer` e as classes treinadas.

O classificador sklearn não é executado no backend ONNX.

## 15. Equivalência

Arquivo de testes:

```text
tests/unit/test_onnx_equivalence.py
```

Critérios:

- mesmas classes;
- mesma ordem das classes;
- mesma classe prevista;
- mesmas labels de probabilidades;
- probabilidades dentro de `atol=1e-5`.

Resultado:

```text
19 testes de equivalência aprovados
```

A arquitetura híbrida foi adotada somente após passar por esse gate.

## 16. FastAPI

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

Contrato de `/predict`:

```text
prediction
probabilities
inference_time_ms
```

O contrato externo permanece estável ao longo das evoluções do projeto.

## 17. Instrumentação Prometheus

Arquivo:

```text
src/medtriage/api/metrics.py
```

Estratégia:

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
métricas
```

Métricas:

```text
medtriage_http_requests_total
medtriage_http_request_duration_seconds
medtriage_http_errors_total
```

Labels:

```text
method
route
status_code
```

Rotas excluídas:

```text
/metrics
/docs
/openapi.json
```

## 18. Prometheus

Docker image:

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

## 19. Grafana

Docker image:

```text
grafana/grafana:11.6.0
```

Datasource:

```text
UID: prometheus
URL: http://prometheus:9090
```

Dashboard:

```text
MedTriage API Monitoring
UID: medtriage-api-monitoring
```

Painéis:

1. Total de Requisições
2. Latência HTTP p95
3. Taxa de Erro

## 20. Docker Compose

Serviços:

```text
api        :8000
prometheus :9090
grafana    :3000
```

Fluxo:

```text
API
 ↓ /metrics
Prometheus
 ↓
Grafana
```

## 21. CI/CD

Arquivo:

```text
.github/workflows/ci.yml
```

Jobs:

```text
quality-and-build
airflow-dag-validation
```

### quality-and-build

Executa:

- checkout;
- Python 3.12.2;
- Poetry 2.4.3;
- instalação;
- Ruff;
- format check;
- pytest;
- geração de modelo sintético;
- Docker build.

### airflow-dag-validation

Executa:

- instalação isolada do Airflow;
- inicialização da base;
- serialização de DAGs;
- verificação de import errors;
- validação da DAG;
- validação das tasks.

## 22. Airflow

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

Configuração:

```text
schedule=None
catchup=False
```

## 23. Benchmark

Arquivos:

```text
src/medtriage/benchmarking/latency.py
src/medtriage/benchmarking/comparison.py
```

Metodologia:

```text
5 textos fixos
20 warm-ups
500 medições
mesma máquina
mesmo escopo model_inference
modelos carregados previamente
```

O benchmark final executa baseline e ONNX na mesma sessão de comparação.

Artefatos:

```text
baseline_latency.json
optimized_latency.json
latency_comparison.json
```

## 24. Resultado final do benchmark

```text
Baseline mean     2.8575 ms
ONNX mean         2.9234 ms
Mean speedup      0.9775x

Baseline p50      2.6761 ms
ONNX p50          2.6615 ms
p50 speedup       1.0055x

Baseline p95      3.9202 ms
ONNX p95          4.2578 ms
p95 speedup       0.9207x
```

Interpretação:

- p50 praticamente equivalente;
- mean levemente pior no ONNX;
- p95 pior no ONNX;
- throughput também levemente inferior.

O backend ONNX não produziu ganho material de performance.

## 25. Por que ONNX não acelerou este caso

A regressão logística original possui baixo custo.

O backend otimizado acrescenta:

```text
CSR sparse
 ↓
float32
 ↓
dense tensor de 186.957 features
 ↓
ONNX Runtime
```

O custo dessa conversão reduz ou elimina o benefício do runtime ONNX para um classificador linear simples.

Isso não invalida a técnica; demonstra que otimização deve ser validada empiricamente no contexto real.

## 26. Estado de qualidade

Última suíte completa observada:

```text
84 passed
```

Também:

```text
ruff check: OK
ruff format --check: OK
```

## 27. Limitações arquiteturais

- proxy acadêmica sem validação clínica;
- dataset não versionado;
- artefatos de modelo não versionados;
- inferência otimizada depende do vectorizer sklearn;
- conversão CSR -> dense possui overhead;
- benchmark depende do hardware local;
- modelo simples limita potencial de ganho com ONNX;
- Airflow é mantido em runtime isolado;
- execução é CPU-only.

## 28. Mapa final de componentes

```text
Dataset
  ↓
Data Loader / Validation
  ↓
Training
  ↓
baseline_pipeline.joblib
  ├───────────────┬──────────────────┐
  ↓               ↓                  ↓
Evaluation     sklearn API       ONNX export
  ↓               ↓                  ↓
evaluation.json  /predict       optimized_model.onnx
                                    ↓
                         sklearn TF-IDF + ONNX LR
                                    ↓
                                benchmark
                                    ↓
                         latency_comparison.json


GitHub Actions ───── qualidade / Docker / Airflow

FastAPI
  ↓
/metrics
  ↓
Prometheus
  ↓
Grafana
```
