# Arquitetura — MedTriage MLOps

## 1. Objetivo

O MedTriage MLOps é um projeto acadêmico para classificação de urgência em textos médicos usando as classes `normal`, `attention` e `urgent`.

Essas classes são uma proxy acadêmica derivada do Medical Abstracts Text Classification Corpus e não representam um protocolo clínico validado.

## 2. Estado atual ao final do Bloco 3

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
      ↓                ↓
validate_artifacts  POST /predict
                       ↓
                    Docker

baseline_pipeline.joblib
          ↓
benchmarking/latency.py
          ↓
baseline_latency.json
```

O endpoint `GET /health` permanece preservado.

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
Docker
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
- medir latência.

## 13. FastAPI

Arquivo:

```text
src/medtriage/api/app.py
```

Endpoints:

```text
GET /health
POST /predict
```

O modelo é carregado no lifespan da aplicação.

## 14. Docker

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

## 15. Artefato temporário para CI

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

## 16. GitHub Actions

Arquivo:

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
Poetry
  ↓
install
  ↓
Ruff
  ↓
pytest
  ↓
prepare CI model
  ↓
Docker build
```

### Job `airflow-dag-validation`

```text
checkout
  ↓
Python 3.12.2
  ↓
Airflow 3.3.1
  ↓
ML dependencies
  ↓
MedTriage editable
  ↓
airflow db migrate
  ↓
airflow dags reserialize
  ↓
import errors
  ↓
DAG validation
  ↓
task validation
```

## 17. Estratégia de dependências do Airflow

Airflow não faz parte do `pyproject.toml` principal.

Motivos:

- não é necessário para a API;
- possui constraints próprias;
- ambiente principal é Windows;
- Airflow é Linux/POSIX-oriented;
- evita conflitos e peso desnecessário.

Arquivo:

```text
airflow/requirements-airflow.txt
```

Conteúdo:

```text
apache-airflow==3.3.1
pandas>=3.0.5,<4.0.0
scikit-learn>=1.9.0,<2.0.0
```

## 18. Runtime Airflow local

Ambiente validado:

```text
WSL2
Ubuntu 24.04 LTS
Python 3.12
Apache Airflow 3.3.1
```

MedTriage é instalado em editable mode:

```bash
python -m pip install -e . --no-deps
```

## 19. DAG de treinamento

Arquivo:

```text
dags/training_pipeline.py
```

Identificador:

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

### `validate_data`

Executa:

```text
load_dataset(TRAIN_DATA_PATH)
load_dataset(TEST_DATA_PATH)
```

### `train_model`

Executa:

```text
run_training()
```

### `evaluate_model`

Executa:

```text
run_evaluation()
```

### `validate_artifacts`

Valida:

```text
MODEL_ARTIFACT_PATH
EVALUATION_ARTIFACT_PATH
```

## 20. Princípio da DAG fina

A DAG contém somente orquestração.

Não duplica:

- transformações;
- target mapping;
- TF-IDF;
- Logistic Regression;
- seed;
- paths;
- persistência;
- métricas.

Regra:

```text
Airflow sabe QUANDO executar.
MedTriage sabe COMO executar.
```

## 21. Execução end-to-end

Comando:

```bash
airflow dags test medtriage_training_pipeline 2026-09-10
```

Resultado:

```text
validate_data       success
train_model         success
evaluate_model      success
validate_artifacts  success
DagRun              success
```

## 22. Testes estruturais da DAG

Arquivo:

```text
tests/unit/test_airflow_dag.py
```

A validação usa `ast` e confirma:

- existência do arquivo;
- tasks esperadas;
- `dag_id`;
- `schedule=None`;
- `catchup=False`;
- chamadas às funções MedTriage;
- cadeia de dependências.

Isso evita instalar Airflow no ambiente Poetry principal.

## 23. Validação real da DAG no CI

O job `airflow-dag-validation` instala Airflow em Ubuntu e valida:

- parsing;
- serialização;
- ausência de import errors;
- `dag_id`;
- quatro tasks.

Assim, teste estático e teste de integração se complementam.

## 24. Benchmark baseline

Arquivo:

```text
src/medtriage/benchmarking/latency.py
```

Resultados:

```text
mean_ms             2.2997
p50_ms              2.2074
p95_ms              2.8114
throughput_req_s  434.84
```

Será usado no Bloco 5 como referência.

## 25. Cobertura ao final do Bloco 3

Suite:

```text
40 testes
```

Validações locais:

```text
pytest
ruff check
ruff format --check
```

Validações remotas:

```text
quality-and-build
airflow-dag-validation
```

## 26. Limitações deliberadas

### DAG manual

`schedule=None` porque não há fonte automática de novos dados nem frequência definida de retreino.

### Dataset fora do Git

O CI não executa treino end-to-end porque os dados reais não são versionados.

### Graphviz

Não foi instalado por não ser necessário para execução.

### MLflow

Não foi incluído por não ser requisito expresso e não ser necessário para o escopo atual.

## 27. Próximos blocos

### Bloco 4

```text
FastAPI
  ↓
/metrics
  ↓
Prometheus
  ↓
Grafana
```

### Bloco 5

```text
baseline sklearn
      ↓
ONNX / quantização
      ↓
benchmark comparativo
      ↓
trade-offs
```

## 28. Evolução da arquitetura

```text
Bloco 1: FastAPI + Docker + estrutura base

Bloco 2: Dataset + NLP + treino + avaliação + inferência + benchmark

Bloco 3: GitHub Actions + Docker CI + Airflow + DAG + testes de orquestração
```
