# MedTriage MLOps

Projeto desenvolvido para o Tech Challenge da Fase 3 da Pós-Tech FIAP.

O objetivo geral é construir uma solução de MLOps para classificação de urgência
em textos médicos, incluindo API de inferência, containerização, automação de
pipelines, observabilidade e otimização de latência.

## Status do projeto

O projeto está no final do BLOCO 1 — Fundação técnica e decisão arquitetural.

Neste estágio já estão implementados:

- estrutura Python em layout `src/`;
- gerenciamento de dependências com Poetry;
- lint e formatação com Ruff;
- testes com pytest;
- configuração centralizada básica;
- logging básico com a biblioteca padrão;
- aplicação mínima com FastAPI;
- endpoint `GET /health`;
- Dockerfile funcional;
- execução da API em container como usuário não-root;
- documentação inicial da arquitetura.

Ainda não fazem parte do estado atual:

- dataset definitivo;
- preprocessing NLP;
- treinamento de modelo;
- endpoint `/predict`;
- persistência de modelo;
- benchmark de latência;
- ONNX;
- Airflow;
- GitHub Actions;
- Prometheus;
- Grafana;
- endpoint `/metrics`.

## Stack

### Implementada no Bloco 1

- Python 3.12.2
- Poetry
- FastAPI
- Uvicorn
- pytest
- Ruff
- Docker

### Planejada para os próximos blocos

- Scikit-learn
- Apache Airflow
- GitHub Actions
- Prometheus
- Grafana
- ONNX

## Estrutura atual

```text
medtriage-mlops/
├── docs/
│   └── architecture.md
├── src/
│   └── medtriage/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       └── api/
│           ├── __init__.py
│           ├── app.py
│           └── schemas.py
├── tests/
│   ├── integration/
│   │   └── test_health.py
│   └── unit/
│       └── test_config.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Instalação

O projeto utiliza Poetry para gerenciamento de dependências.

Instale as dependências com:

```bash
poetry install
```

## Qualidade e testes

Execute os testes:

```bash
poetry run pytest
```

Execute o lint:

```bash
poetry run ruff check .
```

Verifique a formatação:

```bash
poetry run ruff format --check .
```

## API local

Execute a aplicação localmente com:

```bash
poetry run uvicorn medtriage.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Com a API em execução:

```bash
curl http://127.0.0.1:8000/health
```

Resposta esperada:

```json
{"status":"ok"}
```

A documentação interativa do FastAPI fica disponível em:

```text
http://127.0.0.1:8000/docs
```

## Docker

Construa a imagem:

```bash
docker build -t medtriage-mlops .
```

Execute o container:

```bash
docker run --rm -p 8000:8000 --name medtriage-api medtriage-mlops
```

Com o container em execução:

```bash
curl http://127.0.0.1:8000/health
```

Resposta esperada:

```json
{"status":"ok"}
```

A imagem instala apenas dependências de runtime e executa a aplicação como
usuário não-root.

Para validar o usuário:

```bash
docker exec medtriage-api whoami
```

Resposta esperada:

```text
appuser
```

## Decisão arquitetural de inferência

A principal forma de inferência planejada para o sistema é em tempo real.

No cenário proposto, o usuário envia um texto ou laudo médico e espera receber
a classificação de urgência imediatamente. Por isso, a interface principal será
uma API REST.

Processamento em batch continua sendo útil para cenários como:

- reprocessamento de dados históricos;
- preparação de dataset;
- treinamento;
- re-treinamento;
- processamento offline em larga escala.

Entretanto, batch não é a estratégia principal para a inferência operacional.

## Estratégia de cloud

A estratégia recomendada é executar a API containerizada em um serviço gerenciado
de containers com suporte nativo a HTTP e autoscaling.

Como serviço de referência, o projeto considera o Google Cloud Run por sua
simplicidade operacional e boa adequação ao padrão:

```text
Container Docker
       ↓
Serviço gerenciado
       ↓
API HTTP
       ↓
Inferência em tempo real
```

A escolha é arquitetural e acadêmica. Não existe requisito de deploy real em
cloud neste estágio.

A solução permanece portável para serviços equivalentes, como:

- Azure Container Apps;
- AWS App Runner;
- AWS ECS/Fargate.

O uso de Docker reduz o acoplamento ao provedor e favorece reprodutibilidade e
portabilidade entre ambientes.

## Arquitetura alvo

A arquitetura global planejada para o projeto é:

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

GitHub Actions será responsável futuramente por:

- lint;
- testes;
- build.

Detalhes adicionais estão registrados em:

```text
docs/architecture.md
```

## Evolução planejada

### Bloco 2

- dataset;
- preprocessing NLP;
- treinamento;
- persistência de modelo;
- endpoint `/predict`;
- benchmark baseline.

### Bloco 3

- GitHub Actions;
- Airflow.

### Bloco 4

- Prometheus;
- Grafana;
- `/metrics`;
- observabilidade.

### Bloco 5

- ONNX;
- benchmark do modelo otimizado;
- comparação de latência;
- documentação final;
- vídeo STAR.