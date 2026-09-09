# MedTriage MLOps

Projeto desenvolvido para o Tech Challenge da Fase 3 da Pós-Tech FIAP.

O objetivo geral do projeto é construir uma solução de MLOps para classificação
de urgência em textos médicos, incluindo API de inferência, containerização,
automação de pipelines, observabilidade e otimização de latência.

## Status

Em desenvolvimento.

Atualmente o projeto está na etapa de fundação técnica.

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
    └── logging.py

tests/
└── unit/
    └── test_config.py
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