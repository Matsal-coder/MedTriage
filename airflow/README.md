# Airflow orchestration environment

Apache Airflow is used only for training pipeline orchestration.

It is intentionally kept outside the main Poetry environment because:

- the MedTriage application runtime does not require Airflow;
- Apache Airflow recommends installation with its official constraints;
- the project is developed on Windows, while Airflow is executed in a Linux environment through WSL2;
- keeping Airflow isolated avoids inflating or destabilizing the FastAPI runtime dependencies.

## Environment

Validated environment:

- WSL2
- Ubuntu 24.04 LTS
- Python 3.12
- Apache Airflow 3.3.1

The main MedTriage project continues to use Python 3.12.2 through Poetry.

## Create the Airflow environment

Inside Ubuntu/WSL:

```bash
mkdir -p ~/airflow-medtriage
cd ~/airflow-medtriage

python3 -m venv .venv
source .venv/bin/activate
```

If the virtual environment module is unavailable:

```bash
sudo apt update
sudo apt install -y python3.12-venv
```

## Install Airflow

Define the Airflow and Python versions:

```bash
AIRFLOW_VERSION=3.3.1
PYTHON_VERSION=3.12
```

Define the official constraints file:

```bash
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

From the MedTriage repository, install the Airflow requirement using the official constraints:

```bash
python -m pip install \
  -r airflow/requirements-airflow.txt \
  --constraint "${CONSTRAINT_URL}"
```

## Validate the installation

```bash
airflow version
python -c "import airflow; print(airflow.__version__)"
airflow --help
```

Expected Airflow version:

```text
3.3.1
```

## Architecture boundary

Airflow is responsible only for orchestration.

The DAG must reuse the existing application code, especially:

- `medtriage.modeling.train.run_training()`
- `medtriage.modeling.evaluate.run_evaluation()`

Machine learning logic must remain under `src/medtriage/`.

The DAG must not duplicate:

- preprocessing;
- target mapping;
- TF-IDF configuration;
- Logistic Regression configuration;
- random seed;
- artifact paths;
- persistence logic;
- evaluation logic.

The training dataset is not versioned in Git. It must be available locally under `data/raw/` before running the training DAG.
