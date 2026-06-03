"""
preprocessing.py — Feature engineering and data validation pipeline.
Returns both transformed arrays AND the fitted preprocessor for
inference-time consistency (fit on train, transform on test/prod).
"""

import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.validation import check_is_fitted

log = logging.getLogger(__name__)


# ── Validation ───────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, target_column: str) -> None:
    """Raise early with a clear message on common data problems."""
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )
    if df.empty:
        raise ValueError("DataFrame is empty — cannot preprocess.")

    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 50]
    if not high_missing.empty:
        log.warning(
            "Columns with >50%% missing values (will still be imputed): %s",
            high_missing.to_dict(),
        )

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        log.warning("%d duplicate rows detected.", duplicate_count)


# ── Pipeline builder ─────────────────────────────────────────────────────────

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    max_categories: int = 50,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numeric and categorical features.

    Args:
        numeric_features:     List of numeric column names.
        categorical_features: List of categorical column names.
        numeric_strategy:     Imputation strategy for numeric cols.
        categorical_strategy: Imputation strategy for categorical cols.
        max_categories:       Columns with more unique values are dropped
                              from one-hot encoding to avoid explosion.

    Returns:
        An unfitted ColumnTransformer.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_strategy)),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=categorical_strategy)),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,      # always returns a dense array
            max_categories=max_categories,
        )),
    ])

    transformers = []
    if len(numeric_features) > 0:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if len(categorical_features) > 0:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No numeric or categorical features found after column detection.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ── Public API ───────────────────────────────────────────────────────────────

def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    config: dict | None = None,
) -> tuple[np.ndarray, pd.Series, ColumnTransformer]:
    """
    Validate, split features/target, fit and transform in one call.

    Fits the preprocessor on the full df passed in. In your training
    pipeline pass only the training split; in inference pass only the
    test/production data alongside the already-fitted preprocessor via
    transform_data().

    Args:
        df:            Raw input DataFrame.
        target_column: Name of the label column.
        config:        Optional dict with keys:
                         numeric_strategy, categorical_strategy, max_categories

    Returns:
        X_processed:  np.ndarray of shape (n_samples, n_features)
        y:            pd.Series of labels
        preprocessor: Fitted ColumnTransformer (save this for inference)
    """
    cfg = config or {}

    validate_dataframe(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_features     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    log.info(
        "Features detected — numeric: %d, categorical: %d",
        len(numeric_features),
        len(categorical_features),
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_strategy=cfg.get("numeric_strategy", "median"),
        categorical_strategy=cfg.get("categorical_strategy", "most_frequent"),
        max_categories=cfg.get("max_categories", 50),
    )

    X_processed = preprocessor.fit_transform(X)

    # fit_transform can still return sparse on older sklearn builds
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    log.info(
        "Preprocessing complete — output shape: %s, target shape: %s",
        X_processed.shape,
        y.shape,
    )

    return X_processed, y, preprocessor


def transform_data(
    df: pd.DataFrame,
    target_column: str,
    preprocessor: ColumnTransformer,
) -> tuple[np.ndarray, pd.Series]:
    """
    Apply an already-fitted preprocessor to new data (test set or production).

    Never call fit_transform on test data — that leaks statistics.
    Always use this function for anything outside the training split.

    Args:
        df:           Raw input DataFrame (test or production).
        target_column: Name of the label column.
        preprocessor: A fitted ColumnTransformer from preprocess_data().

    Returns:
        X_transformed: np.ndarray
        y:             pd.Series
    """
    check_is_fitted(preprocessor)
    validate_dataframe(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    return X_transformed, y