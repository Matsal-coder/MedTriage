# Arquitetura — MedTriage MLOps

## 1. Objetivo do sistema

O MedTriage MLOps é um projeto acadêmico voltado à construção de uma solução de
classificação de urgência em textos médicos.

O objetivo final é receber um texto ou laudo e classificá-lo em uma categoria de
urgência, por exemplo:

- normal;
- atenção;
- urgente.

A solução completa deverá contemplar não apenas o modelo de Machine Learning,
mas também os componentes necessários para disponibilização, automação,
observabilidade e otimização da inferência.

## 2. Estado atual da arquitetura

Ao final do BLOCO 1, o projeto possui uma fundação mínima e funcional formada por:

```text
Docker
  ↓
Uvicorn
  ↓
FastAPI
  ↓
GET /health
```

Os componentes implementados até este estágio são:

- estrutura de pacote Python em layout `src/`;
- configuração centralizada básica;
- logging básico;
- FastAPI;
- endpoint de health check;
- testes automatizados;
- Dockerfile;
- execução não-root no container.

Nenhum modelo de Machine Learning está carregado neste momento.

## 3. Arquitetura alvo do projeto

A arquitetura global planejada é:

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

GitHub Actions atuará de forma transversal no repositório para validar:

```text
Git push / Pull Request
          ↓
     GitHub Actions
          ↓
       Ruff
          ↓
       pytest
          ↓
    Docker build
```

Essa é a arquitetura alvo. Os componentes ainda não implementados serão
adicionados somente nos blocos correspondentes.

## 4. Componentes atuais

### 4.1 `src/medtriage/config.py`

Centraliza configurações básicas da aplicação.

No Bloco 1 contém valores como:

- nome da aplicação;
- versão;
- nível padrão de logging.

O objetivo é impedir que constantes centrais sejam duplicadas em diferentes
partes do projeto.

Nos próximos blocos poderá evoluir para receber configurações relacionadas a
caminhos de artefatos, parâmetros e ambiente.

### 4.2 `src/medtriage/logging.py`

Centraliza a inicialização do logging da aplicação.

A implementação utiliza a biblioteca padrão `logging`, evitando dependências
adicionais nesta fase.

O logging atual é deliberadamente simples. Logging estruturado, correlação de
requisições e integrações externas não fazem parte do Bloco 1.

### 4.3 `src/medtriage/api/app.py`

É o entrypoint atual da aplicação FastAPI.

Responsabilidades atuais:

- instanciar a aplicação;
- consumir nome e versão da configuração;
- inicializar logging;
- registrar `GET /health`.

O objeto exposto é:

```text
medtriage.api.app:app
```

Esse contrato será utilizado por Uvicorn e Docker e não deve ser quebrado sem
necessidade.

No Bloco 2, esse módulo deverá evoluir para registrar a rota de inferência.

### 4.4 `src/medtriage/api/schemas.py`

Contém contratos de entrada e saída da API.

No Bloco 1 existe apenas o schema de resposta do health check.

No Bloco 2 deverão ser adicionados os schemas necessários à inferência,
preferencialmente sem misturar lógica de modelo com definição de contratos HTTP.

### 4.5 `GET /health`

O endpoint:

```text
GET /health
```

retorna:

```json
{"status":"ok"}
```

Sua finalidade é indicar que o processo da API está ativo e acessível.

No Bloco 1 ele não verifica disponibilidade de modelo, dataset ou dependências
externas, pois esses componentes ainda não existem.

### 4.6 Docker

A aplicação é empacotada em imagem baseada em:

```text
python:3.12.2-slim
```

O container:

- instala Poetry;
- utiliza `poetry.lock`;
- instala apenas dependências principais;
- copia `src/`;
- executa Uvicorn;
- expõe a porta 8000;
- roda como `appuser`, sem privilégios de root.

O entrypoint lógico do runtime é:

```text
Docker
  ↓
uvicorn
  ↓
medtriage.api.app:app
```

## 5. Fluxo atual

O fluxo real existente ao final do Bloco 1 é:

```text
Cliente
  ↓
HTTP :8000
  ↓
Docker
  ↓
Uvicorn
  ↓
src/medtriage/api/app.py
  ↓
GET /health
  ↓
HealthResponse
```

A configuração e o logging são carregados durante a inicialização da aplicação:

```text
config.py ────────┐
                  ↓
              api/app.py
                  ↑
logging.py ───────┘
```

## 6. Evolução prevista

### Bloco 2 — Dataset, NLP e modelo

O Bloco 2 deverá introduzir:

- dataset definitivo;
- validação e carregamento dos dados;
- preprocessing NLP;
- treinamento;
- avaliação;
- persistência do modelo;
- carregamento do artefato;
- endpoint `/predict`;
- benchmark baseline de latência.

A FastAPI existente deve ser reutilizada, não substituída.

### Bloco 3 — CI/CD e Airflow

O Bloco 3 deverá introduzir:

- workflows do GitHub Actions;
- lint automatizado;
- testes automatizados;
- build automatizado;
- DAGs do Airflow;
- pipeline de treinamento e/ou automação prevista no desafio.

### Bloco 4 — Observabilidade

O Bloco 4 deverá introduzir:

- Prometheus;
- endpoint `/metrics`;
- métricas de aplicação;
- métricas relacionadas à inferência;
- Grafana;
- dashboards.

### Bloco 5 — Otimização

O Bloco 5 deverá introduzir:

- exportação ou conversão para ONNX;
- benchmark do modelo otimizado;
- comparação entre baseline e versão otimizada;
- documentação consolidada;
- material para apresentação e vídeo STAR.

## 7. Decisão de inferência

### Batch

Inferência em batch é adequada quando um conjunto de registros pode ser processado
de forma assíncrona ou periódica.

No contexto deste projeto, batch pode ser útil para:

- reprocessar textos históricos;
- gerar previsões offline;
- preparar avaliações;
- executar pipelines de treinamento;
- realizar tarefas periódicas.

### Real-time

Na operação principal do sistema, um texto médico é submetido para triagem e o
resultado precisa estar disponível imediatamente.

Esse comportamento se encaixa melhor em inferência síncrona, exposta através de
API REST.

Fluxo esperado:

```text
Texto
  ↓
POST /predict
  ↓
Modelo
  ↓
Classificação
  ↓
Resposta HTTP
```

### Escolha

A arquitetura principal será real-time.

Batch permanece como estratégia complementar para pipelines e processamento
offline.

## 8. Estratégia de cloud

A recomendação arquitetural é executar a API em um serviço gerenciado de
containers.

O padrão desejado é:

```text
Imagem Docker
     ↓
Serviço de containers
     ↓
Endpoint HTTP
     ↓
Autoscaling
```

### Serviço de referência

Como referência arquitetural, considera-se o Google Cloud Run.

A escolha se deve principalmente a:

- execução direta de containers;
- suporte nativo a HTTP;
- escalabilidade gerenciada;
- baixa necessidade de administração de infraestrutura;
- boa adequação a uma API stateless;
- simplicidade compatível com o escopo acadêmico do Tech Challenge.

O projeto não depende tecnicamente do Cloud Run.

Serviços equivalentes poderiam hospedar a mesma imagem, como:

- Azure Container Apps;
- AWS App Runner;
- AWS ECS/Fargate.

Não há implementação real de cloud no Bloco 1.

## 9. Portabilidade com Docker

Docker estabelece uma fronteira clara entre a aplicação e a infraestrutura.

A mesma imagem conceitualmente pode ser executada:

```text
Notebook local
     │
     ├── Docker Desktop
     │
     ├── Cloud Run
     │
     ├── Azure Container Apps
     │
     └── ECS/Fargate
```

Isso reduz diferenças entre ambientes e evita acoplamento desnecessário ao
provedor de cloud.

## 10. Decisões arquiteturais do Bloco 1

### Python

Decisão:

```text
Python 3.12.2
```

Motivo:

- versão moderna;
- compatibilidade com a stack escolhida;
- alinhamento entre ambiente local e Docker.

### Gerenciamento de dependências

Decisão:

```text
Poetry
```

Motivo:

- centralização em `pyproject.toml`;
- lockfile;
- grupos de dependências;
- reprodutibilidade.

### Layout de pacote

Decisão:

```text
src/medtriage/
```

Motivo:

- separação clara entre código e raiz;
- melhor comportamento de imports;
- estrutura preparada para expansão.

### Qualidade

Decisão:

```text
pytest + Ruff
```

Motivo:

- ferramentas simples;
- boa integração;
- baixo overhead;
- Ruff cobre lint e formatação.

### Configuração

Decisão:

```text
config.py simples
```

Não foi adicionado um framework de settings no Bloco 1 porque ainda não existem
configurações externas suficientes para justificar essa camada.

### Logging

Decisão:

```text
logging da biblioteca padrão
```

Motivo:

- nenhuma dependência adicional;
- suficiente para o estágio atual.

### API

Decisão:

```text
FastAPI + Uvicorn
```

Motivo:

- interface HTTP adequada a inferência;
- tipagem e schemas;
- documentação OpenAPI automática;
- baixo custo de implementação.

### Docker

Decisão:

```text
python:3.12.2-slim
```

com instalação somente das dependências de runtime e execução por usuário não-root.

## 11. Contratos que não devem ser quebrados sem necessidade

Os próximos blocos devem preservar, salvo justificativa técnica:

### Entry point

```text
medtriage.api.app:app
```

### Health check

```text
GET /health
```

Resposta mínima:

```json
{"status":"ok"}
```

### Layout Python

```text
src/medtriage/
```

Novos módulos devem preferencialmente ser adicionados dentro deste namespace.

### Configuração centralizada

Valores globais não devem ser espalhados em arquivos diferentes se pertencem à
configuração da aplicação.

### Testes

Toda nova funcionalidade com comportamento verificável deve ser acompanhada por
teste adequado.

### Docker

O container deve permanecer executável sem privilégios de root, salvo necessidade
técnica devidamente documentada.

## 12. Limites atuais

O Bloco 1 estabelece apenas a fundação.

Não existe neste estágio:

```text
POST /predict
modelo treinado
dataset
pipeline de treinamento
Airflow
CI/CD
Prometheus
Grafana
ONNX
benchmark
```

Essas ausências são deliberadas e fazem parte da divisão arquitetural do projeto,
não representam falhas do Bloco 1.