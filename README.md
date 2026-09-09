# MedTriage MLOps

Projeto desenvolvido para o Tech Challenge da Fase 3 da Pós-Tech FIAP.

O objetivo geral do projeto é construir uma solução de MLOps para classificação
de urgência em textos médicos, incluindo API de inferência, containerização,
automação de pipelines, observabilidade e otimização de latência.

## Status

Em desenvolvimento.

Atualmente o projeto possui:
- fundação técnica em Python;
- gerenciamento de dependências com Poetry;
- validações com pytest e Ruff;
- API mínima com FastAPI;
- endpoint `GET /health`;
- suporte inicial a execução em Docker.

## Stack planejada

- Python
- FastAPI
- Docker
- GitHub Actions
- Apache Airflow
- Prometheus
- Grafana
- Scikit-learn
- ONNX

A implementação será realizada de forma incremental.

## Estrutura atual

```text
src/
└── medtriage/
    ├── __init__.py
    ├── config.py
    ├── logging.py
    └── api/
        ├── __init__.py
        ├── app.py
        └── schemas.py

tests/
├── unit/
│   └── test_config.py
└── integration/
    └── test_health.py
```

## Desenvolvimento

O projeto utiliza Poetry para gerenciamento de dependências.

Instale as dependências com:

```bash
poetry install
```

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

## API

Execute a aplicação localmente com:

```bash
poetry run uvicorn medtriage.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Com a API em execução, valide o endpoint de saúde:

```bash
curl http://127.0.0.1:8000/health
```

Resposta esperada:

```json
{"status":"ok"}
```

A documentação interativa gerada pelo FastAPI fica disponível em:

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

Com o container em execução, verifique:

```bash
curl http://127.0.0.1:8000/health
```

Resposta esperada:

```json
{"status":"ok"}
```

A imagem é construída apenas com as dependências de runtime e executa a API
como usuário não-root.
