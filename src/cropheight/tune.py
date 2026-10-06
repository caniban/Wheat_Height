from __future__ import annotations

from typing import Any, Dict

import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from .features import build_feature_selection_pipeline

def objective_for_model(
    X,
    y,
    model_name: str,
    fs_method: str,
    n_features: int,
    trial: optuna.Trial,
) -> float:
    if model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [10, 20, 30, None]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10]),
        }
    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7]),
            "subsample": trial.suggest_categorical("subsample", [0.8, 1.0]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 1.0]),
        }
    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
            "num_leaves": trial.suggest_categorical("num_leaves", [30, 50, 70]),
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    estimator = __import__("src.cropheight.models", fromlist=["get_model"]).get_model(model_name)
    estimator.set_params(**params)

    pipeline = build_feature_selection_pipeline(estimator, fs_method, n_features)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)

    return score.mean()