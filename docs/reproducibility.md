# Guia de reprodução a partir de clone limpo

## 1. Objetivo

Este documento descreve como reproduzir o projeto MedTriage MLOps a partir de
um clone limpo do repositório.

O fluxo cobre:

```text
clone
  ↓
ambiente Python + Poetry
  ↓
dataset
  ↓
validação do dataset por SHA-256
  ↓
testes e qualidade
  ↓
treino do baseline
  ↓
avaliação
  ↓
diagnóstico de qualidade de dados
  ↓
export ONNX
  ↓
benchmark comparativo
  ↓
Docker
  ↓
Docker Compose
  ↓
API
  ↓
Prometheus
  ↓
Grafana
```

O objetivo é reproduzir a entrega técnica do Tech Challenge, e não reconstruir
um ambiente clínico real.

> O MedTriage utiliza uma proxy acadêmica de triagem. As classes `normal`,
> `attention` e `urgent` não representam protocolo clínico validado.

## 2. Repositório

Repositório oficial do projeto:

```text
https://github.com/Matsal-coder/MedTriage.git
```

Clone:

```bash
git clone https://github.com/Matsal-coder/MedTriage.git
cd MedTriage
```

Para confirmar o commit que está sendo reproduzido:

```bash
git rev-parse HEAD
git status -sb
```

Para uma reprodução formal, registre o SHA do commit utilizado junto dos
resultados obtidos.

## 3. Pré-requisitos

### Ambiente principal

O projeto utiliza:

```text
Python >=3.12,<3.13
Poetry
Docker Desktop / Docker Engine
Docker Compose
Git
```

A versão utilizada durante o desenvolvimento foi:

```text
Python 3.12.2
Poetry 2.4.3
```

O projeto foi desenvolvido principalmente em Windows, com Git Bash para os
comandos do repositório.

### Airflow

O Apache Airflow é mantido em ambiente separado do Poetry principal.

Durante o desenvolvimento local foi utilizado:

```text
WSL2
Ubuntu 24.04 LTS
Apache Airflow 3.3.1
```

O Airflow não é necessário para executar a API, o benchmark de latência ou a
stack de observabilidade via Docker Compose.

## 4. Instalação das dependências

Na raiz do repositório:

```bash
poetry install
```

Confirme o ambiente:

```bash
poetry run python --version
```

O esperado é uma versão Python 3.12 compatível com:

```text
>=3.12,<3.13
```

## 5. Dataset

O projeto utiliza o:

```text
Medical Abstracts Text Classification Corpus
```

Fonte pública:

```text
https://github.com/sebischair/Medical-Abstracts-TC-Corpus
```

Os arquivos utilizados são:

```text
medical_tc_train.csv
medical_tc_test.csv
medical_tc_labels.csv
```

Eles não são versionados no repositório MedTriage.

### 5.1 Obtenção dos arquivos

Uma forma reproduzível de obter os arquivos no Git Bash é:

```bash
git clone --depth 1 \
  https://github.com/sebischair/Medical-Abstracts-TC-Corpus.git \
  .dataset-source

mkdir -p data/raw

cp .dataset-source/medical_tc_train.csv data/raw/
cp .dataset-source/medical_tc_test.csv data/raw/
cp .dataset-source/medical_tc_labels.csv data/raw/

rm -rf .dataset-source
```

Ao final:

```bash
ls -lh data/raw/
```

O diretório deve conter os três CSVs.

### 5.2 Verificação por SHA-256

Os arquivos efetivamente utilizados no desenvolvimento do MedTriage possuem os
seguintes hashes SHA-256:

```text
medical_tc_train.csv
ad53aebc682d6b87a5647f619a079bb446d286fdc93bf0159b812418f5758609

medical_tc_test.csv
1eecea73c9ecad292c55e10403bd139fab9580545d6878482997c5564d51ac05

medical_tc_labels.csv
8a27ae03339c798103678efa8012f744a723ff71a80f2b2c1355ee249564adc5
```

Verificação no Git Bash:

```bash
sha256sum \
  data/raw/medical_tc_train.csv \
  data/raw/medical_tc_test.csv \
  data/raw/medical_tc_labels.csv
```

Se os hashes forem diferentes, o conteúdo do dataset não é byte a byte igual
ao utilizado nesta entrega.

Isso não significa necessariamente que o arquivo esteja inválido, mas impede
afirmar reprodução exata do mesmo conjunto de dados sem investigar a diferença.

## 6. Estrutura esperada do dataset

Os arquivos devem estar em:

```text
data/raw/
├── medical_tc_labels.csv
├── medical_tc_test.csv
└── medical_tc_train.csv
```

O pipeline espera, entre outras, as colunas:

```text
medical_abstract
condition_label
```

O campo:

```text
triage_label
```

é criado pelo próprio projeto durante a preparação.

O mapeamento acadêmico é:

```text
1 -> urgent
2 -> attention
3 -> attention
4 -> urgent
5 -> normal
```

## 7. Validação inicial do projeto

Antes de treinar qualquer modelo, execute:

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
```

Uma reprodução limpa deve terminar sem falhas.

A quantidade total de testes pode aumentar ao longo do desenvolvimento. O
critério de sucesso é a suíte completa aprovada, não um número rígido de testes.

## 8. Treinamento do baseline

Execute:

```bash
poetry run python -m medtriage.modeling.train
```

O fluxo utiliza:

```text
medical_tc_train.csv
        ↓
load_dataset()
        ↓
validate_dataset()
        ↓
add_triage_labels()
        ↓
split_training_data()
        ↓
TF-IDF + LogisticRegression
```

O split treino/validação é reproduzível e utiliza:

```text
RANDOM_SEED = 837
```

O modelo treinado é persistido localmente em:

```text
artifacts/models/baseline_pipeline.joblib
```

Esse artefato é intermediário e não é versionado no Git.

Confirme sua criação:

```bash
ls -lh artifacts/models/baseline_pipeline.joblib
```

## 9. Avaliação do baseline

Com o baseline treinado:

```bash
poetry run python -m medtriage.modeling.evaluate
```

A avaliação utiliza:

- validação derivada do conjunto oficial de treino;
- conjunto oficial de teste;
- a mesma preparação de dados usada no treinamento.

O artefato de avaliação é gerado localmente em:

```text
artifacts/models/evaluation.json
```

Ele é intermediário e permanece ignorado pelo Git.

## 10. Diagnóstico de qualidade de dados

Para reproduzir a análise pós-auditoria de duplicação, sobreposição entre
conjuntos e múltiplos rótulos:

```bash
poetry run python -m medtriage.data.diagnostics
```

Artefato:

```text
artifacts/evaluation/data_overlap_analysis.json
```

Esse JSON é versionado como evidência reproduzível.

Os principais valores esperados para o dataset utilizado nesta entrega são:

```text
validation rows whose text exists in effective train = 677
shared unique texts                                   = 659

official test rows whose text exists in effective train = 846
shared unique texts                                      = 828
```

A interpretação completa está em:

```text
docs/data-quality.md
```

Essas sobreposições devem ser tratadas como uma limitação de independência das
amostras. O projeto não conclui, de forma categórica, que houve inflação das
métricas.

## 11. Exportação do modelo para ONNX

Após gerar o baseline:

```bash
poetry run python -m medtriage.modeling.onnx_export
```

O modelo otimizado é salvo em:

```text
artifacts/models/optimized_model.onnx
```

Esse arquivo é versionado no repositório como evidência da entrega final.

A implementação possui testes automatizados de equivalência entre o pipeline
sklearn e o backend ONNX.

## 12. Benchmark comparativo de latência

Com baseline e ONNX disponíveis:

```bash
poetry run python -m medtriage.benchmarking.comparison
```

A comparação produz:

```text
artifacts/benchmarks/baseline_latency.json
artifacts/benchmarks/optimized_latency.json
artifacts/benchmarks/latency_comparison.json
```

Os três JSONs finais são versionados.

O benchmark final é comparativo e sequencial na mesma sessão, utilizando os
mesmos textos, warm-up e número de execuções para ambos os backends.

Ele não deve ser interpretado como um teste de carga HTTP nem como isolamento
puro dos runtimes.

Para metodologia e resultados:

```text
docs/latency-comparison.md
```

## 13. Execução local da API

Depois de treinar o baseline:

```bash
poetry run uvicorn medtriage.api.app:app \
  --host 127.0.0.1 \
  --port 8000
```

Endpoints principais:

```text
GET  /health
POST /predict
GET  /metrics
```

Em outro terminal, valide o health check:

```bash
curl http://127.0.0.1:8000/health
```

Valide as métricas:

```bash
curl http://127.0.0.1:8000/metrics
```

Para uma inferência:

```bash
curl -X POST \
  http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"severe chest pain and shortness of breath."}'
```

A classe retornada é apenas a saída da proxy acadêmica do projeto.

## 14. Docker

### 14.1 Build da imagem

Na raiz do projeto:

```bash
docker build -t medtriage .
```

### 14.2 Execução isolada da API

```bash
docker run --rm \
  -p 8000:8000 \
  medtriage
```

Valide:

```bash
curl http://127.0.0.1:8000/health
```

## 15. Docker Compose e observabilidade

Para construir e iniciar a stack:

```bash
docker compose up --build
```

Serviços esperados:

```text
API        -> http://localhost:8000
Prometheus -> http://localhost:9090
Grafana    -> http://localhost:3000
```

Confirme:

```bash
docker compose ps
```

### API

```bash
curl http://localhost:8000/health
```

### Prometheus

Abra:

```text
http://localhost:9090
```

O Prometheus coleta:

```text
http://api:8000/metrics
```

dentro da rede do Compose.

### Grafana

Abra:

```text
http://localhost:3000
```

A fonte Prometheus e o dashboard do MedTriage são provisionados
automaticamente pelos arquivos do projeto.

## 16. Gerar tráfego para observabilidade

Com a stack ativa, o projeto disponibiliza o gerador de requisições.

Exemplo:

```bash
poetry run python scripts/generate_requests.py \
  --requests 50 \
  --delay 0.05 \
  --invalid-ratio 0
```

Também é possível gerar uma parcela de requisições inválidas para observar
erros nas métricas.

Exemplo:

```bash
poetry run python scripts/generate_requests.py \
  --requests 200 \
  --delay 0.02 \
  --invalid-ratio 0.1
```

Após gerar tráfego, verifique o Prometheus e o dashboard do Grafana.

## 17. Airflow

O Airflow utiliza ambiente isolado do Poetry principal.

Os arquivos relevantes são:

```text
airflow/requirements-airflow.txt
dags/training_pipeline.py
```

O DAG reutiliza as funções Python de treinamento e avaliação do projeto.

A validação do DAG também é executada pelo GitHub Actions em job separado,
evitando acoplar as dependências do Airflow ao ambiente principal.

Para reprodução local do Airflow, utilize WSL2/Ubuntu e siga as instruções
específicas de:

```text
airflow/README.md
```

## 18. CI

O GitHub Actions está definido em:

```text
.github/workflows/ci.yml
```

O pipeline valida, entre outros pontos:

- testes automatizados;
- Ruff;
- build da aplicação;
- validação do Airflow em ambiente isolado.

O CI utiliza um modelo sintético quando necessário para validar componentes em
runner limpo sem depender do dataset real não versionado.

## 19. Artefatos versionados e locais

### Versionados

```text
artifacts/models/optimized_model.onnx

artifacts/benchmarks/baseline_latency.json
artifacts/benchmarks/optimized_latency.json
artifacts/benchmarks/latency_comparison.json

artifacts/evaluation/data_overlap_analysis.json
```

### Gerados localmente e não versionados

```text
data/raw/

artifacts/models/baseline_pipeline.joblib
artifacts/models/evaluation.json
```

A separação permite manter a entrega leve sem perder as principais evidências
da otimização, do benchmark e da auditoria de qualidade de dados.

## 20. Sequência mínima de reprodução

Depois de clonar o repositório e colocar o dataset correto em `data/raw/`, a
sequência principal é:

```bash
poetry install

poetry run pytest
poetry run ruff check .
poetry run ruff format --check .

poetry run python -m medtriage.modeling.train
poetry run python -m medtriage.modeling.evaluate
poetry run python -m medtriage.data.diagnostics
poetry run python -m medtriage.modeling.onnx_export
poetry run python -m medtriage.benchmarking.comparison

docker build -t medtriage .
docker compose up --build
```

Em seguida, valide:

```text
http://localhost:8000/health
http://localhost:8000/metrics
http://localhost:9090
http://localhost:3000
```

## 21. Critérios de sucesso

Uma reprodução é considerada funcional quando:

- as dependências são instaladas sem erro;
- os hashes do dataset correspondem aos arquivos documentados, quando o
  objetivo for reprodução exata;
- a suíte de testes passa;
- Ruff passa;
- o baseline é gerado;
- a avaliação é executada;
- o diagnóstico de dados é reproduzido;
- o ONNX é exportado;
- os testes de equivalência permanecem aprovados;
- o benchmark comparativo é executado;
- a imagem Docker é construída;
- a API responde em `/health`;
- `/predict` realiza inferência;
- `/metrics` expõe métricas Prometheus;
- Prometheus coleta a API;
- Grafana inicia com provisionamento automático.

## 22. Limitações de reprodutibilidade

Resultados de latência podem variar entre máquinas, sistemas operacionais e
carga do host.

Por esse motivo:

- os JSONs de benchmark versionados representam a execução final de referência;
- uma nova execução não precisa reproduzir os tempos em milissegundos
  exatamente;
- a metodologia e as conclusões relativas devem ser preservadas;
- não se deve selecionar ou descartar medições apenas para produzir resultados
  mais favoráveis.

Da mesma forma, mudanças futuras no repositório público do dataset podem
alterar os arquivos disponibilizados. Os SHA-256 documentados são a referência
dos arquivos efetivamente utilizados nesta entrega.

## 23. Documentação relacionada

```text
README.md
docs/architecture.md
docs/baseline-results.md
docs/data-quality.md
docs/latency-comparison.md
```
