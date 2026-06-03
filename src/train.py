"""
train.py — Auto model comparison pipeline
Trains all models from registry, logs to MLflow, saves the best.
"""

import logging
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from mlflow.models.signature import infer_signature

from sklearn.pipeline import Pipeline as SklearnPipeline

from src.preprocessing import preprocess_data, transform_data
from src.model_factory import get_models
from src.config_loader import load_config

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    """Return a flat dict of classification metrics."""
    avg = "weighted"
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
    }


def log_run(model_name: str, model, params: dict, metrics: dict,
            X_train, y_pred, dataset_version: str) -> None:
    """Log a single training run to MLflow."""
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)


def print_summary(results: dict) -> None:
    """Print a ranked comparison table of all model results."""
    ranked = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    header = f"{'Rank':<5} {'Model':<25} {'Accuracy':>9} {'F1':>8} {'Precision':>10} {'Recall':>8}"
    log.info("\n" + "=" * len(header))
    log.info(header)
    log.info("-" * len(header))
    for rank, (name, r) in enumerate(ranked, 1):
        m = r["metrics"]
        marker = " ◀ BEST" if rank == 1 else ""
        log.info(
            f"{rank:<5} {name:<25} {m['accuracy']:>9.4f} {m['f1']:>8.4f}"
            f" {m['precision']:>10.4f} {m['recall']:>8.4f}{marker}"
        )
    log.info("=" * len(header))


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    # 1. Config
    config = load_config()
    train_cfg = config["train"]
    dataset_version = config.get("dataset", {}).get("version", "unknown")

    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "auto_model_comparison"))

    # 2. Data — split RAW df first, then fit preprocessor on train only
    df = pd.read_csv(config["dataset"]["path"])
    target = config["target_column"]

    df_train, df_test = train_test_split(
        df,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
    )

    # fit_transform on train → no leakage of test statistics
    X_train, y_train, preprocessor = preprocess_data(
        df_train, target, config=config.get("preprocessing")
    )
    # transform-only on test using the already-fitted preprocessor
    X_test, y_test = transform_data(df_test, target, preprocessor)

    log.info("Dataset split: %d train / %d test samples", len(X_train), len(X_test))

    # 3. Train all models
    models = get_models(config["models"])
    results = {}

    for model_name, model in models.items():
        log.info("Training %s ...", model_name)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        params  = model.get_params()

        log_run(model_name, model, params, metrics, X_train, y_pred, dataset_version)
        results[model_name] = {"model": model, "metrics": metrics, "accuracy": metrics["accuracy"]}

        log.info("  accuracy=%.4f  f1=%.4f", metrics["accuracy"], metrics["f1"])

    # 4. Select best
    best_name  = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    best_acc   = results[best_name]["accuracy"]

    print_summary(results)
    log.info("Best model: %s  (accuracy=%.4f)", best_name, best_acc)

    # 5. Bundle preprocessor + best model into one artifact, then save
    #    At inference: pipeline.predict(raw_df) handles everything end-to-end
    full_pipeline = SklearnPipeline([
        ("preprocessor", preprocessor),
        ("model",        best_model),
    ])

    output_path = Path(config.get("output", {}).get("model_path", "models/pipeline.pkl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_pipeline, output_path)
    log.info("Full pipeline (preprocessor + model) saved → %s", output_path)


if __name__ == "__main__":
    run_pipeline()