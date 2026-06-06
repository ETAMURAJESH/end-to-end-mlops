"""
train.py — Universal auto model comparison pipeline.
Works on ANY dataset: classification or regression.
Auto-detects task type from preprocessing, picks correct models + metrics.
"""

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


# ── Helpers ───────────────────────────────────────────────────────────────────


def primary_metric(task_type: str) -> str:
    """The single metric used to rank and pick the best model."""
    return "accuracy" if task_type == "classification" else "r2"


def compute_metrics(y_true, y_pred, task_type: str) -> dict:
    """Return the right metrics depending on task type."""
    if task_type == "classification":
        avg = "weighted"
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1": round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
            "precision": round(
                precision_score(y_true, y_pred, average=avg, zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_true, y_pred, average=avg, zero_division=0), 4
            ),
        }
    else:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "r2": round(r2_score(y_true, y_pred), 4),
            "mae": round(mean_absolute_error(y_true, y_pred), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
        }


def cross_validate(model, X_train, y_train, task_type: str, cv: int = 5) -> dict:
    """k-fold CV on the training set — the honest accuracy number."""
    scoring = "accuracy" if task_type == "classification" else "r2"
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    return {
        "cv_mean": round(scores.mean(), 4),
        "cv_std": round(scores.std(), 4),
        "cv_min": round(scores.min(), 4),
        "cv_max": round(scores.max(), 4),
    }


# ── MLflow logging ────────────────────────────────────────────────────────────


def log_run(
    model_name: str,
    model,
    params: dict,
    metrics: dict,
    cv_metrics: dict,
    X_train,
    y_pred,
    dataset_version: str,
) -> None:
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(cv_metrics)
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)


# ── Summary + leakage check ───────────────────────────────────────────────────


def print_summary(results: dict, task_type: str) -> None:
    """Ranked table: test metric + CV mean ± std for every model."""
    ranked = sorted(
        results.items(), key=lambda x: x[1]["cv_metrics"]["cv_mean"], reverse=True
    )

    if task_type == "classification":
        header = (
            f"{'Rank':<5} {'Model':<25} {'Test Acc':>9} {'CV Mean':>8} "
            f"{'Std':>6} {'F1':>7} {'Precision':>10} {'Recall':>8}"
        )
    else:
        header = (
            f"{'Rank':<5} {'Model':<25} {'Test R2':>8} "
            f"{'CV Mean':>8} {'Std':>6} {'MAE':>8} {'RMSE':>8}"
        )

    sep = "=" * len(header)
    log.info("\n%s\n%s\n%s", sep, header, "-" * len(header))

    for rank, (name, r) in enumerate(ranked, 1):
        m = r["metrics"]
        cv = r["cv_metrics"]
        marker = "  BEST" if rank == 1 else ""

        if task_type == "classification":
            log.info(
                "%d     %-25s %9.4f %8.4f  %6.4f %7.4f %10.4f %8.4f%s",
                rank,
                name,
                m["accuracy"],
                cv["cv_mean"],
                cv["cv_std"],
                m["f1"],
                m["precision"],
                m["recall"],
                marker,
            )
        else:
            log.info(
                "%d     %-25s %8.4f %8.4f  %6.4f %8.4f %8.4f%s",
                rank,
                name,
                m["r2"],
                cv["cv_mean"],
                cv["cv_std"],
                m["mae"],
                m["rmse"],
                marker,
            )

    log.info(sep)


def check_for_leakage(results: dict, task_type: str) -> None:
    """Warn if scores look suspiciously perfect."""
    metric_key = primary_metric(task_type)
    for name, r in results.items():
        test_score = r["metrics"][metric_key]
        cv_score = r["cv_metrics"]["cv_mean"]
        if test_score >= 0.99 and task_type == "classification":
            log.warning(
                "%s test accuracy=%.4f is suspiciously high. "
                "Check Remaining features in preprocessing log.",
                name,
                test_score,
            )
        if abs(test_score - cv_score) > 0.10:
            log.warning(
                "%s test %s (%.4f) vs CV mean (%.4f) gap > 10%%. "
                "Model may be overfitting.",
                name,
                metric_key,
                test_score,
                cv_score,
            )


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_pipeline() -> None:

    # 1. Config
    config = load_config()
    train_cfg = config["train"]
    dataset_version = config.get("dataset", {}).get("version", "unknown")
    cv_folds = train_cfg.get("cv_folds", 5)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    log.info("MLflow tracking URI: %s", tracking_uri)

    mlflow.set_experiment(
        config.get("mlflow", {}).get("experiment_name", "auto_model_comparison")
    )

    # 2. Load raw CSV
    df = pd.read_csv(config["dataset"]["path"])
    log.info("Raw dataset shape: %s  columns: %s", df.shape, df.columns.tolist())

    # 3. Preprocess
    X, y, preprocessor, task_type = preprocess_data(
        df,
        config["target_column"],
        config=config.get("preprocessing"),
    )
    log.info("Task type detected: %s", task_type)

    # 4. Train/test split
    split_kwargs = dict(
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
    )
    if task_type == "classification":
        split_kwargs["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
    log.info("Split — train: %d  test: %d", len(X_train), len(X_test))
    if task_type == "classification":
        log.info("Class balance (train): %s", y_train.value_counts().to_dict())

    # 5. Get models
    models = get_models(config["models"], task_type=task_type)
    log.info("Models to train: %s", list(models.keys()))

    # 6. Train all models
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
            "  %s=%.4f  cv_mean=%.4f±%.4f",
            primary_metric(task_type),
            metrics[primary_metric(task_type)],
            cv_metrics["cv_mean"],
            cv_metrics["cv_std"],
        )

    # 7. Summary + leakage check
    print_summary(results, task_type)
    check_for_leakage(results, task_type)

    # 8. Pick best by CV mean
    best_name = max(results, key=lambda k: results[k]["cv_metrics"]["cv_mean"])
    best_model = results[best_name]["model"]
    best_cv = results[best_name]["cv_metrics"]["cv_mean"]

    log.info("Best model: %s  (cv_mean=%.4f)", best_name, best_cv)

    best_preds = best_model.predict(X_test)
    if task_type == "classification":
        log.info(
            "\nClassification Report - %s:\n%s",
            best_name,
            classification_report(y_test, best_preds),
        )
        log.info("Confusion Matrix:\n%s", confusion_matrix(y_test, best_preds))

    # 9. Save pipeline
    full_pipeline = SklearnPipeline(
        [("preprocessor", preprocessor), ("model", best_model)]
    )

    output_path = Path(
        config.get("output", {}).get("model_path", "models/pipeline.pkl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_pipeline, output_path)
    log.info("Pipeline (preprocessor + %s) saved -> %s", best_name, output_path)


if __name__ == "__main__":
    run_pipeline()
