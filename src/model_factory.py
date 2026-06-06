"""
model_factory.py — Universal model registry.
Returns classification OR regression models based on task type.
Adding a new model = one line in the registry dict.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

# ── Registry ──────────────────────────────────────────────────────────────────
# Each entry: "model_key": (ModelClass, "classification" | "regression" | "both")

_REGISTRY = {
    # ── Classification ────────────────────────────────────────────────────────
    "logistic_regression": (LogisticRegression, "classification"),
    "decision_tree": (DecisionTreeClassifier, "classification"),
    "random_forest": (RandomForestClassifier, "classification"),
    "knn": (KNeighborsClassifier, "classification"),
    "svm": (SVC, "classification"),
    "naive_bayes": (GaussianNB, "classification"),
    "gradient_boosting": (GradientBoostingClassifier, "classification"),
    # ── Regression ────────────────────────────────────────────────────────────
    "linear_regression": (LinearRegression, "regression"),
    "ridge": (Ridge, "regression"),
    "lasso": (Lasso, "regression"),
    "decision_tree_regressor": (DecisionTreeRegressor, "regression"),
    "random_forest_regressor": (RandomForestRegressor, "regression"),
    "knn_regressor": (KNeighborsRegressor, "regression"),
    "svr": (SVR, "regression"),
    "gradient_boosting_regressor": (GradientBoostingRegressor, "regression"),
}


def get_model(name: str, params: dict | None = None):
    """Instantiate a single model by registry key."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. " f"Available: {list(_REGISTRY.keys())}"
        )
    model_cls, _ = _REGISTRY[name]
    return model_cls(**(params or {}))


def get_models(model_names: list[str], task_type: str = "classification") -> dict:
    """
    Return instantiated models filtered to the given task type.

    Args:
        model_names: List of model keys from config["models"].
                     Pass ["auto"] to use ALL models for the task type.
        task_type:   "classification" or "regression"

    Returns:
        dict of {model_name: fitted_model_instance}
    """
    if model_names == ["auto"]:
        # Use every registered model that matches the task type
        model_names = [key for key, (_, t) in _REGISTRY.items() if t == task_type]

    models = {}
    for name in model_names:
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}"
            )
        model_cls, model_task = _REGISTRY[name]
        if model_task != task_type:
            raise ValueError(
                f"Model '{name}' is for {model_task} "
                f"but your dataset needs {task_type}. "
                f"Fix config['models'] or use ['auto']."
            )
        models[name] = model_cls()

    return models
