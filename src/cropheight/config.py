from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

@dataclass
class ExperimentConfig:
    seed: int = 42
    cv_folds: int = 5
    n_features_to_select: int = 19
    data_path: str = "data/crop_heights.xlsx"
    results_dir: str = "results"
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    experiment_name: str = "Crop_Height_SAR_Estimation"
    feature_selection_methods: List[str] = field(default_factory=lambda: ["ALL", "MI", "LASSO", "RFE"])
    models: List[str] = field(default_factory=lambda: ["RandomForest", "XGBoost", "LightGBM"])
    ablation_sets: List[Dict[str, Any]] = field(default_factory=list)
    param_grids: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    test_size: float = 0.2

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        config = cls(
            seed=payload.get("seed", 42),
            cv_folds=payload.get("cv_folds", 5),
            n_features_to_select=payload.get("n_features_to_select", 19),
            data_path=payload.get("data_path", "data/crop_heights.xlsx"),
            results_dir=payload.get("results_dir", "results"),
            mlflow_tracking_uri=payload.get("mlflow_tracking_uri", "sqlite:///mlruns.db"),
            experiment_name=payload.get("experiment_name", "Crop_Height_SAR_Estimation"),
            feature_selection_methods=payload.get("feature_selection_methods", ["ALL", "MI", "LASSO", "RFE"]),
            models=payload.get("models", ["RandomForest", "XGBoost", "LightGBM"]),
            ablation_sets=payload.get("ablation_sets", []),
            param_grids=payload.get("param_grids", {}),
            test_size=payload.get("test_size", 0.2),
        )

        if not config.ablation_sets:
            config.ablation_sets = [
                {"name": "Only C-Band (Sentinel-1)", "columns_prefix": ["S1"]},
                {"name": "Only X-Band (PAZ & TDX)", "columns_prefix": ["PAZ_", "TDX_"]},
                {"name": "Combined (C-Band + X-Band)", "columns_prefix": ["S1", "PAZ_", "TDX_"]},
            ]

        if not 0 < config.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        return config