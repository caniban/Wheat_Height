from __future__ import annotations

from typing import Dict, Any

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

MODEL_LIBRARY = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=30,
        boosting_type="gbdt",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
}

PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [30, 50, 70],
        "boosting_type": ["gbdt", "dart"]
    }
}

def get_model(name: str):
    if name not in MODEL_LIBRARY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_LIBRARY[name]

def get_param_grid(name: str) -> Dict[str, Any]:
    if name not in PARAM_GRIDS:
        raise ValueError(f"Unknown parameter grid: {name}")
    return PARAM_GRIDS[name]