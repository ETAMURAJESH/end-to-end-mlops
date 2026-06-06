"""
preprocessing.py — Universal feature engineering pipeline.
Auto-detects task type (classification vs regression),
drops junk columns (IDs, free text, high-cardinality),
and returns transformed arrays + fitted preprocessor.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

log = logging.getLogger(__name__)

# Columns that are almost always junk — override via config if needed
_AUTO_DROP_PATTERNS = [
    "id",
    "passengerid",
    "customerid",
    "userid",
    "rowid",  # IDs
    "name",
    "firstname",
    "lastname",
    "fullname",  # free text names
    "ticket",
    "cabin",
    "address",
    "email",
    "phone",  # high-cardinality strings
]

# Threshold: if a categorical column has more unique values than this
# fraction of total rows, it's likely an ID or free-text → auto-drop
_HIGH_CARDINALITY_RATIO = 0.5


# ── Auto-detectors ────────────────────────────────────────────────────────────


def detect_task_type(y: pd.Series) -> str:
    """
    Infer whether this is a classification or regression task.

    Rules:
      - dtype object/category      → classification
      - ≤ 20 unique integer values → classification
      - float or many unique ints  → regression
    """
    if y.dtype == "object" or str(y.dtype) == "category":
        return "classification"
    n_unique = y.nunique()
    if pd.api.types.is_integer_dtype(y) and n_unique <= 20:
        return "classification"
    return "regression"


def auto_drop_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    """
    Automatically identify columns that should be dropped before training.

    Drops if ANY of these conditions are true:
      1. Column name matches a known junk pattern (case-insensitive)
      2. Categorical column with unique ratio > 0.5 (likely an ID / free text)
      3. Only 1 unique value (zero variance — carries no information)
    """
    to_drop = []
    n_rows = len(df)

    for col in df.columns:
        if col == target_column:
            continue

        col_lower = col.lower()

        # Rule 1 — name pattern match
        if any(pat in col_lower for pat in _AUTO_DROP_PATTERNS):
            log.info("Auto-drop (name pattern): %s", col)
            to_drop.append(col)
            continue

        # Rule 2 — high cardinality categorical
        if df[col].dtype == "object":
            ratio = df[col].nunique() / n_rows
            if ratio > _HIGH_CARDINALITY_RATIO:
                log.info("Auto-drop (high cardinality %.2f): %s", ratio, col)
                to_drop.append(col)
                continue

        # Rule 3 — zero variance
        if df[col].nunique() <= 1:
            log.info("Auto-drop (zero variance): %s", col)
            to_drop.append(col)

    return to_drop


# ── Validation ────────────────────────────────────────────────────────────────


def validate_dataframe(df: pd.DataFrame, target_column: str) -> None:
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found. "
            f"Available: {df.columns.tolist()}"
        )
    if df.empty:
        raise ValueError("DataFrame is empty — cannot preprocess.")

    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 50]
    if not high_missing.empty:
        log.warning(
            "Columns with >50%% missing values (still imputed): %s",
            high_missing.to_dict(),
        )

    dupes = df.duplicated().sum()
    if dupes > 0:
        log.warning("%d duplicate rows detected.", dupes)


# ── Pipeline builder ──────────────────────────────────────────────────────────


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    max_categories: int = 50,
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=numeric_strategy)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=categorical_strategy)),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=max_categories,
                ),
            ),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No usable features found after dropping junk columns.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ── Public API ────────────────────────────────────────────────────────────────


def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    config: dict | None = None,
) -> tuple[np.ndarray, pd.Series, ColumnTransformer, str]:
    """
    Full preprocessing for ANY CSV dataset.

    Auto-detects:
      - Task type  (classification / regression)
      - Junk columns to drop (IDs, free text, high-cardinality)

    Override auto-detection via config:
      config = {
          "task_type":            "classification",   # override auto-detect
          "drop_columns":         ["col1", "col2"],   # extra columns to drop
          "keep_columns":         ["col3"],           # force-keep even if auto wants to drop
          "numeric_strategy":     "median",
          "categorical_strategy": "most_frequent",
          "max_categories":       50,
      }

    Returns:
        X_processed:  np.ndarray
        y:            pd.Series
        preprocessor: Fitted ColumnTransformer
        task_type:    "classification" or "regression"
    """
    cfg = config or {}

    validate_dataframe(df, target_column)

    y = df[target_column]

    # ── Task type ─────────────────────────────────────────────────────────────
    task_type = cfg.get("task_type") or detect_task_type(y)
    log.info("Task type: %s", task_type)

    # ── Drop columns ──────────────────────────────────────────────────────────
    auto_drop = auto_drop_columns(df, target_column)
    manual_drop = cfg.get("drop_columns", [])
    keep = set(cfg.get("keep_columns", []))

    # merge auto + manual, then remove anything in keep_columns
    all_drop = list(set(auto_drop + manual_drop + [target_column]) - keep)

    X = df.drop(columns=all_drop, errors="ignore")
    log.info("Dropped  : %s", [c for c in all_drop if c != target_column])
    log.info("Remaining: %s", X.columns.tolist())

    # ── Feature type detection ────────────────────────────────────────────────
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    log.info(
        "Numeric (%d): %s  |  Categorical (%d): %s",
        len(numeric_features),
        numeric_features,
        len(categorical_features),
        categorical_features,
    )

    # ── Build + fit ───────────────────────────────────────────────────────────
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_strategy=cfg.get("numeric_strategy", "median"),
        categorical_strategy=cfg.get("categorical_strategy", "most_frequent"),
        max_categories=cfg.get("max_categories", 50),
    )

    X_processed = preprocessor.fit_transform(X)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    log.info(
        "Preprocessing done — X: %s  y: %s  task: %s",
        X_processed.shape,
        y.shape,
        task_type,
    )

    return X_processed, y, preprocessor, task_type
