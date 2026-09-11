# MedTriage MLOps

Projeto desenvolvido para o Tech Challenge da Fase 3 da Pós-Tech FIAP.

O objetivo geral é construir uma solução de MLOps para classificação acadêmica de urgência em textos médicos, incluindo API de inferência, containerização, automação de pipelines, observabilidade e otimização de latência.

IMPORTANTE:
O mapeamento para as classes `normal`, `attention` e `urgent` é uma simplificação acadêmica criada para demonstrar a arquitetura MLOps. Ele NÃO representa uma regra clínica validada e NÃO deve ser interpretado como sistema médico de triagem real.

## Status do projeto

O projeto está no final do BLOCO 3 — CI/CD e orquestração com Airflow.

Neste estágio já estão implementados:

- estrutura Python em layout `src/`;
- gerenciamento de dependências com Poetry;
- lint e formatação com Ruff;
- testes com pytest;
- configuração centralizada;
- logging básico;
- aplicação FastAPI;
- endpoints `GET /health` e `POST /predict`;
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
- execução end-to-end da DAG localmente.

Ainda não fazem parte do estado atual:

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
- Poetry 2.4.3
- FastAPI
- Uvicorn
- pandas
- scikit-learn
- Joblib
- pytest
- Ruff
- Docker
- GitHub Actions
- Apache Airflow 3.3.1
- WSL2 / Ubuntu 24.04 LTS para Airflow local

### Planejada

- Prometheus
- Grafana
- ONNX
- quantização

## Estrutura atual

```text
medtriage-mlops/
├── .github/workflows/ci.yml
├── airflow/
│   ├── README.md
│   └── requirements-airflow.txt
├── dags/
│   └── training_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── architecture.md
│   └── baseline-results.md
├── src/medtriage/
│   ├── api/
│   ├── benchmarking/
│   ├── ci/
│   ├── data/
│   ├── modeling/
│   ├── config.py
│   └── logging.py
├── tests/
│   ├── integration/
│   └── unit/
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

Ao final do Bloco 3, a suite possui 40 testes.

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

```bash
poetry run uvicorn medtriage.api.app:app --host 127.0.0.1 --port 8000
```

Endpoints:

```text
GET /health
POST /predict
```

## Docker

```bash
docker build -t medtriage-mlops .
docker run --rm -p 8000:8000 medtriage-mlops
```

O container executa como usuário não-root.

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

### Job `airflow-dag-validation`

```text
checkout
 ↓
Python 3.12.2
 ↓
Airflow 3.3.1
 ↓
dependências ML
 ↓
MedTriage editable
 ↓
airflow db migrate
 ↓
airflow dags reserialize
 ↓
list-import-errors
 ↓
validar DAG
 ↓
validar tasks
```

Os dois jobs foram validados com sucesso em pull request.

## Airflow

A configuração detalhada está em:

```text
airflow/README.md
```

Arquitetura local:

```text
Windows + Poetry
└── aplicação MedTriage

WSL2 / Ubuntu
└── Apache Airflow 3.3.1
    └── DAG MedTriage
```

O pacote é disponibilizado ao ambiente Airflow em editable mode:

```bash
python -m pip install -e . --no-deps
```

## DAG de treinamento

Arquivo:

```text
dags/training_pipeline.py
```

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

A DAG reutiliza `load_dataset()`, `run_training()` e `run_evaluation()` e não duplica lógica de ML.

## Teste local da DAG

```bash
airflow dags test medtriage_training_pipeline 2026-09-10
```

Execução validada:

```text
validate_data       success
train_model         success
evaluate_model      success
validate_artifacts  success
DagRun              success
```

## Testes da DAG

Arquivo:

```text
tests/unit/test_airflow_dag.py
```

Usa `ast` para validar o contrato estrutural sem instalar Airflow no Poetry principal.

São validados:

- existência da DAG;
- quatro task functions;
- `dag_id`;
- `schedule=None`;
- `catchup=False`;
- reutilização das funções MedTriage;
- ordem das dependências.

A integração real é validada pelo job `airflow-dag-validation`.

## Benchmark baseline

```bash
poetry run python -m medtriage.benchmarking.latency
```

Resultados:

```text
mean_ms             2.2997
p50_ms              2.2074
p95_ms              2.8114
throughput_req_s  434.84
```

Esses valores serão usados como referência no Bloco 5.

## Estado ao final do Bloco 3

```text
Git push / Pull Request
          ↓
   GitHub Actions
     ↙         ↘
quality       Airflow
and build     validation

Dataset
  ↓
Airflow
  ↓
validate_data
  ↓
run_training()
  ↓
baseline_pipeline.joblib
  ↓
run_evaluation()
  ↓
evaluation.json
  ↓
validate_artifacts

baseline_pipeline.joblib
    ↙              ↘
 FastAPI        benchmark
```

## Próximos passos

### Bloco 4 — Observabilidade

- `/metrics`;
- Prometheus;
- Grafana;
- métricas técnicas e de ML;
- dashboards.

### Bloco 5 — Otimização

- ONNX;
- quantização se aplicável;
- benchmark comparativo;
- análise de trade-offs;
- documentação final;
- vídeo STAR.

## Aviso de uso

Este projeto é exclusivamente acadêmico. As classes de triagem não foram validadas clinicamente e não devem ser utilizadas para decisões médicas reais.
