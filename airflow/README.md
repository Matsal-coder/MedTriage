# Airflow orchestration

Airflow is used only to orchestrate the MedTriage training and evaluation workflow.

The MedTriage application itself continues to use Poetry and the existing project
structure. Airflow is kept in a separate Linux/WSL environment because:

- the FastAPI runtime does not require Airflow;
- Apache Airflow installation is designed around pip/uv with official constraints;
- the local development host is Windows, while Airflow is Linux/POSIX-oriented;
- keeping the environments separate avoids inflating or destabilizing the application runtime.

## Validated local environment

The Airflow orchestration environment was validated with:

- WSL2
- Ubuntu 24.04 LTS
- Python 3.12
- Apache Airflow 3.3.1

The main MedTriage project continues to use Python 3.12.2 through Poetry.

## Create the Airflow environment

Inside WSL:

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

Then recreate the environment:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

## Install Airflow

Define the Airflow and Python versions:

```bash
AIRFLOW_VERSION=3.3.1
PYTHON_VERSION=3.12
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

From the MedTriage repository, install the Airflow orchestration dependencies:

```bash
python -m pip install \
  -r airflow/requirements-airflow.txt \
  --constraint "${CONSTRAINT_URL}"
```

The Airflow-specific requirements are:

```text
apache-airflow==3.3.1
pandas>=3.0.5,<4.0.0
scikit-learn>=1.9.0,<2.0.0
```

`joblib` does not need to be declared separately because it is installed as a
dependency of scikit-learn.

## Install MedTriage in the Airflow environment

From the MedTriage repository:

```bash
python -m pip install -e . --no-deps
```

The Airflow environment installs its own orchestration dependencies from
`airflow/requirements-airflow.txt`, while the MedTriage source package is linked in
editable mode.

This allows Airflow DAGs to import:

- `medtriage.data.loader`
- `medtriage.modeling.train`
- `medtriage.modeling.evaluate`

without duplicating the application logic.

## Validate the installation

Check the Airflow installation:

```bash
airflow version
python -c "import airflow; print(airflow.__version__)"
airflow --help
```

Check the MedTriage package:

```bash
python -c "import medtriage; print(medtriage.__file__)"
```

Check the machine learning dependencies:

```bash
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import sklearn; print('sklearn:', sklearn.__version__)"
python -c "import joblib; print('joblib:', joblib.__version__)"
```

Check the imports used by the DAG:

```bash
python - <<'PY'
from medtriage.data.loader import load_dataset
from medtriage.modeling.train import run_training
from medtriage.modeling.evaluate import run_evaluation

print("MedTriage imports OK")
print(load_dataset)
print(run_training)
print(run_evaluation)
PY
```

## Airflow configuration

The Airflow metadata and runtime files are kept outside the repository:

```bash
export AIRFLOW_HOME="$HOME/airflow-medtriage/airflow-home"
```

Point Airflow to the repository DAG directory:

```bash
export AIRFLOW__CORE__DAGS_FOLDER="/path/to/MedTriage/dags"
```

Disable the example DAGs:

```bash
export AIRFLOW__CORE__LOAD_EXAMPLES=False
```

Create the Airflow home directory:

```bash
mkdir -p "$AIRFLOW_HOME"
```

Confirm the configuration:

```bash
echo "$AIRFLOW_HOME"
echo "$AIRFLOW__CORE__DAGS_FOLDER"
airflow config get-value core dags_folder
```

## Initialize the metadata database

Run:

```bash
airflow db migrate
```

## DAG architecture

The DAG is defined in:

```text
dags/training_pipeline.py
```

Its orchestration flow is:

```text
validate_data
    ↓
train_model
    ↓
evaluate_model
    ↓
validate_artifacts
```

The DAG is intentionally thin and declarative.

Airflow is responsible only for orchestration.

The machine learning logic remains inside `src/medtriage/`.

The DAG reuses the existing project functions:

```text
medtriage.modeling.train.run_training()
medtriage.modeling.evaluate.run_evaluation()
```

It must not duplicate:

- dataset preprocessing;
- triage label mapping;
- TF-IDF configuration;
- Logistic Regression configuration;
- random seed configuration;
- artifact paths;
- model persistence;
- evaluation logic.

The DAG also reuses the centralized project configuration and dataset loader.

## Data requirements

The training and evaluation DAG expects the real Medical Abstracts TC Corpus files to
already exist under:

```text
data/raw/
```

Expected files:

```text
data/raw/medical_tc_train.csv
data/raw/medical_tc_test.csv
```

The DAG does not download the dataset.

If the required data is absent, the validation task must fail clearly.

## DAG schedule

The training DAG uses:

```text
schedule=None
```

This means the DAG is manually triggered.

No arbitrary retraining schedule is introduced because the project does not currently
define an automatic source of new training data or a required retraining frequency.

The DAG also uses:

```text
catchup=False
```

to avoid retroactive runs.

## Validate DAG discovery

Force Airflow to parse and serialize the project DAG:

```bash
airflow dags reserialize
```

Check that the DAG is available:

```bash
airflow dags list
```

Check import errors:

```bash
airflow dags list-import-errors
```

Check the DAG tasks:

```bash
airflow tasks list medtriage_training_pipeline
```

Expected tasks:

```text
validate_data
train_model
evaluate_model
validate_artifacts
```

## Execute the DAG locally

Run the DAG end-to-end:

```bash
airflow dags test medtriage_training_pipeline 2026-09-10
```

A successful execution validates the complete orchestration path:

```text
validate_data
    ↓
train_model
    ↓
evaluate_model
    ↓
validate_artifacts
```

The expected result is a successful DAG run with all four tasks completed.

## Notes about example DAGs

When `AIRFLOW__CORE__LOAD_EXAMPLES=False` is set, Airflow disables the built-in example
DAG bundles.

If example DAGs were previously serialized in the metadata database, running:

```bash
airflow dags reserialize
```

refreshes the registered DAGs.

Warnings related only to built-in example DAGs are not part of the MedTriage pipeline.

## Graphviz warning

Airflow may show a warning similar to:

```text
Could not import graphviz. Rendering graph to the graphical format will not be possible.
```

Graphviz is not required for the MedTriage pipeline execution.

The warning affects graphical DAG rendering only and does not prevent:

- DAG parsing;
- DAG serialization;
- task listing;
- DAG execution;
- model training;
- model evaluation.

## Architecture boundary

The intended separation is:

```text
Windows + Poetry
└── MedTriage application development
    ├── FastAPI
    ├── tests
    ├── training logic
    └── evaluation logic

WSL / Linux
└── Airflow runtime
    └── MedTriage training DAG
        ├── validate_data
        ├── run_training()
        ├── run_evaluation()
        └── validate_artifacts
```

This keeps the orchestration layer separate from the application runtime while reusing
the same MedTriage source code.
