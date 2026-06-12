"""
train.py — Universal auto model comparison pipeline.
Works on ANY dataset: classification or regression.

Usage:
    # Use config.yaml defaults
    python -m src.train

    # Override dataset and target at runtime
    python -m src.train --dataset data/iris.csv --target species
    python -m src.train --dataset data/house_prices.csv --target SalePrice
"""

import argparse
import os
import logging
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from mlflow.models.signature import infer_signature

from src.preprocessing import preprocess_data
from src.model_factory import get_models
from src.config_loader import load_config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Train all models on any CSV dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to CSV file. Overrides config.yaml dataset.path.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name. Overrides config.yaml target_column.",
    )
    return parser.parse_args()


# ── Auto-resolvers ────────────────────────────────────────────────────────────


def resolve_dataset(config: dict, cli_dataset: str | None) -> str:
    """
    Resolve dataset path from CLI arg or config.
    Raises clear error if both are 'auto' or missing.
    """
    path = cli_dataset or config.get("dataset", {}).get("path", "auto")
    if path == "auto" or not path:
        raise ValueError(
            "Dataset path not set.\n"
            "Either:\n"
            "  1. Run with --dataset path/to/file.csv\n"
            "  2. Set dataset.path in config.yaml"
        )
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def resolve_target(config: dict, cli_target: str | None, df: pd.DataFrame) -> str:
    """
    Resolve target column from CLI arg or config.
    Raises clear error listing available columns if not set.
    """
    target = cli_target or config.get("target_column", "auto")
    if target == "auto" or not target:
        raise ValueError(
            "Target column not set.\n"
            "Either:\n"
            f"  1. Run with --target column_name\n"
            f"  2. Set target_column in config.yaml\n"
            f"Available columns: {df.columns.tolist()}"
        )
    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found.\n" f"Available columns: {df.columns.tolist()}"
        )
    return target


def resolve_experiment_name(config: dict, dataset_path: str) -> str:
    """Use dataset filename as experiment name when set to auto."""
    name = config.get("mlflow", {}).get("experiment_name", "auto")
    if name == "auto":
        name = Path(dataset_path).stem  # e.g. "tested" or "iris"
    return name


def resolve_drop_columns(config: dict) -> list:
    """Return empty list when drop_columns is auto — auto-drop handles it."""
    drop = config.get("preprocessing", {}).get("drop_columns", "auto")
    if drop == "auto" or drop is None:
        return []  # preprocessing.py auto_drop_columns() handles detection
    return drop


# ── Metrics ───────────────────────────────────────────────────────────────────


def primary_metric(task_type: str) -> str:
    return "accuracy" if task_type == "classification" else "r2"


def compute_metrics(y_true, y_pred, task_type: str) -> dict:
    if task_type == "classification":
        avg = "weighted"
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1": round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
            "precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
        }
    else:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "r2": round(r2_score(y_true, y_pred), 4),
            "mae": round(mean_absolute_error(y_true, y_pred), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
        }


def cross_validate(model, X_train, y_train, task_type: str, cv: int = 5) -> dict:
    scoring = "accuracy" if task_type == "classification" else "r2"
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    return {
        "cv_mean": round(scores.mean(), 4),
        "cv_std": round(scores.std(), 4),
        "cv_min": round(scores.min(), 4),
        "cv_max": round(scores.max(), 4),
    }


# ── MLflow ────────────────────────────────────────────────────────────────────


def log_run(
    model_name,
    model,
    params,
    metrics,
    cv_metrics,
    X_train,
    y_pred,
    dataset_version,
):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(cv_metrics)
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)


# ── Summary ───────────────────────────────────────────────────────────────────


def print_summary(results: dict, task_type: str) -> None:
    ranked = sorted(
        results.items(),
        key=lambda x: x[1]["cv_metrics"]["cv_mean"],
        reverse=True,
    )
    if task_type == "classification":
        header = (
            f"{'Rank':<5} {'Model':<25} {'Test Acc':>9} " f"{'CV Mean':>8} {'Std':>6} {'F1':>7}"
        )
    else:
        header = (
            f"{'Rank':<5} {'Model':<25} {'Test R2':>8} " f"{'CV Mean':>8} {'Std':>6} {'MAE':>8}"
        )
    sep = "=" * len(header)
    log.info("\n%s\n%s\n%s", sep, header, "-" * len(header))
    for rank, (name, r) in enumerate(ranked, 1):
        m = r["metrics"]
        cv = r["cv_metrics"]
        marker = "  BEST" if rank == 1 else ""
        if task_type == "classification":
            log.info(
                "%d     %-25s %9.4f %8.4f  %6.4f %7.4f%s",
                rank,
                name,
                m["accuracy"],
                cv["cv_mean"],
                cv["cv_std"],
                m["f1"],
                marker,
            )
        else:
            log.info(
                "%d     %-25s %8.4f %8.4f  %6.4f %8.4f%s",
                rank,
                name,
                m["r2"],
                cv["cv_mean"],
                cv["cv_std"],
                m["mae"],
                marker,
            )
    log.info(sep)


def check_for_leakage(results: dict, task_type: str) -> None:
    metric_key = primary_metric(task_type)
    for name, r in results.items():
        test_score = r["metrics"][metric_key]
        cv_score = r["cv_metrics"]["cv_mean"]
        if test_score >= 0.99 and task_type == "classification":
            log.warning(
                "%s test accuracy=%.4f is suspiciously high. " "Check for leaked columns.",
                name,
                test_score,
            )
        if abs(test_score - cv_score) > 0.10:
            log.warning(
                "%s test %s (%.4f) vs CV mean (%.4f) gap > 10%%.",
                name,
                metric_key,
                test_score,
                cv_score,
            )


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_pipeline(cli_dataset: str | None = None, cli_target: str | None = None) -> None:

    # 1. Config
    config = load_config()
    train_cfg = config["train"]
    dataset_version = config.get("dataset", {}).get("version", "unknown")
    cv_folds = train_cfg.get("cv_folds", 5)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(tracking_uri)

    # 2. Resolve auto values
    dataset_path = resolve_dataset(config, cli_dataset)
    df = pd.read_csv(dataset_path)
    log.info("Dataset: %s  shape: %s", dataset_path, df.shape)

    target_column = resolve_target(config, cli_target, df)
    log.info("Target column: %s", target_column)

    experiment_name = resolve_experiment_name(config, dataset_path)
    mlflow.set_experiment(experiment_name)
    log.info("MLflow experiment: %s", experiment_name)

    # Merge resolved drop_columns into preprocessing config
    preprocessing_cfg = dict(config.get("preprocessing") or {})
    preprocessing_cfg["drop_columns"] = resolve_drop_columns(config)

    # 3. Preprocess
    X, y, preprocessor, task_type = preprocess_data(df, target_column, config=preprocessing_cfg)
    log.info("Task type: %s", task_type)

    # 4. Split
    split_kwargs = dict(
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
    )
    if task_type == "classification":
        split_kwargs["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
    log.info("Split — train: %d  test: %d", len(X_train), len(X_test))

    # 5. Train all models
    models = get_models(config["models"], task_type=task_type)
    log.info("Models: %s", list(models.keys()))

    results = {}
    for model_name, model in models.items():
        log.info("Training %s ...", model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, task_type)
        cv_metrics = cross_validate(model, X_train, y_train, task_type, cv=cv_folds)
        params = model.get_params()
        log_run(
            model_name,
            model,
            params,
            metrics,
            cv_metrics,
            X_train,
            y_pred,
            dataset_version,
        )
        results[model_name] = {
            "model": model,
            "metrics": metrics,
            "cv_metrics": cv_metrics,
        }
        log.info(
            "  %s=%.4f  cv=%.4f±%.4f",
            primary_metric(task_type),
            metrics[primary_metric(task_type)],
            cv_metrics["cv_mean"],
            cv_metrics["cv_std"],
        )

    # 6. Summary
    print_summary(results, task_type)
    check_for_leakage(results, task_type)

    # 7. Best model
    best_name = max(results, key=lambda k: results[k]["cv_metrics"]["cv_mean"])
    best_model = results[best_name]["model"]
    log.info("Best model: %s  cv_mean=%.4f", best_name, results[best_name]["cv_metrics"]["cv_mean"])

    if task_type == "classification":
        best_preds = best_model.predict(X_test)
        log.info(
            "\nClassification Report:\n%s",
            classification_report(y_test, best_preds),
        )
        log.info("Confusion Matrix:\n%s", confusion_matrix(y_test, best_preds))

    # 8. Save pipeline
    full_pipeline = SklearnPipeline([("preprocessor", preprocessor), ("model", best_model)])
    output_path = Path(config.get("output", {}).get("model_path", "models/pipeline.pkl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_pipeline, output_path)
    log.info("Pipeline saved -> %s", output_path)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(cli_dataset=args.dataset, cli_target=args.target)
