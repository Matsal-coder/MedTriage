"""Static contract tests for the MedTriage Airflow DAG."""

import ast
from pathlib import Path

DAG_PATH = Path("dags/training_pipeline.py")

EXPECTED_TASKS = {
    "validate_data",
    "train_model",
    "evaluate_model",
    "validate_artifacts",
}


def extract_shift_chain_names(node: ast.expr) -> list[str]:
    """Extract variable names from a right-shift dependency chain."""
    if isinstance(node, ast.Name):
        return [node.id]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
        return extract_shift_chain_names(node.left) + extract_shift_chain_names(
            node.right
        )

    return []


def load_dag_tree() -> ast.Module:
    """Parse the DAG source without requiring Airflow in the Poetry environment."""
    source = DAG_PATH.read_text(encoding="utf-8")
    return ast.parse(source)


def test_training_dag_exists() -> None:
    """The training DAG file must be versioned in the expected location."""
    assert DAG_PATH.exists()


def test_training_dag_has_expected_task_functions() -> None:
    """The DAG must define the expected orchestration tasks."""
    tree = load_dag_tree()

    task_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Name) and decorator.id == "task"
            for decorator in node.decorator_list
        )
    }

    assert task_functions == EXPECTED_TASKS


def test_training_dag_uses_expected_dag_id() -> None:
    """The DAG identifier must remain stable for orchestration."""
    tree = load_dag_tree()

    dag_ids = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue

            if not isinstance(decorator.func, ast.Name):
                continue

            if decorator.func.id != "dag":
                continue

            for keyword in decorator.keywords:
                if keyword.arg == "dag_id" and isinstance(
                    keyword.value,
                    ast.Constant,
                ):
                    dag_ids.append(keyword.value.value)

    assert dag_ids == ["medtriage_training_pipeline"]


def test_training_dag_is_manually_triggered_without_catchup() -> None:
    """The DAG must remain manual and must not create retroactive runs."""
    tree = load_dag_tree()

    schedule_values = []
    catchup_values = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Name) or node.func.id != "dag":
            continue

        for keyword in node.keywords:
            if keyword.arg == "schedule":
                schedule_values.append(keyword.value)

            if keyword.arg == "catchup":
                catchup_values.append(keyword.value)

    assert len(schedule_values) == 1
    assert isinstance(schedule_values[0], ast.Constant)
    assert schedule_values[0].value is None

    assert len(catchup_values) == 1
    assert isinstance(catchup_values[0], ast.Constant)
    assert catchup_values[0].value is False


def test_training_dag_reuses_medtriage_pipeline_functions() -> None:
    """The DAG must orchestrate existing ML functions instead of duplicating them."""
    tree = load_dag_tree()

    called_functions = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "load_dataset" in called_functions
    assert "run_training" in called_functions
    assert "run_evaluation" in called_functions


def test_training_dag_has_expected_task_dependencies() -> None:
    """The DAG tasks must preserve the expected execution order."""
    tree = load_dag_tree()

    dependency_chains = [
        extract_shift_chain_names(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.RShift)
    ]

    assert [
        "data_validation",
        "training",
        "evaluation",
        "artifact_validation",
    ] in dependency_chains
