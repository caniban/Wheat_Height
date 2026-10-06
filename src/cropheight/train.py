from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import json

from .config import ExperimentConfig
from .data import get_ablation_features, load_dataset
from .features import build_feature_selection_pipeline, export_multicollinearity_report, get_selected_features
from .models import get_model, get_param_grid

def run_training(config: ExperimentConfig, retrain: bool = True):
    del retrain  # This command explicitly runs a new training experiment.

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(config.data_path)
    y = df["Crop_Height"]
    row_indices = np.arange(len(df))

    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=config.test_size,
        random_state=config.seed,
        shuffle=True,
    )

    evaluation_dir = results_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "test_indices.json").write_text(
        json.dumps(test_indices.tolist()),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    summary_rows = []

    for ablation_config in config.ablation_sets:
        ablation_name = ablation_config["name"]
        ablation_slug = ablation_name.replace(" ", "_")
        feature_columns = get_ablation_features(df, ablation_config)

        if not feature_columns:
            print(f"Skipping ablation with no features: {ablation_name}")
            continue

        X = df[feature_columns]
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        for model_name in config.models:
            for fs_method in config.feature_selection_methods:
                n_features = min(len(feature_columns), config.n_features_to_select)

                pipeline = build_feature_selection_pipeline(
                    get_model(model_name),
                    fs_method,
                    n_features,
                )
                param_grid = {
                    f"model__{name}": values
                    for name, values in get_param_grid(model_name).items()
                }

                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    scoring="neg_root_mean_squared_error",
                    cv=KFold(
                        n_splits=config.cv_folds,
                        shuffle=True,
                        random_state=config.seed,
                    ),
                    refit=True,
                    n_jobs=-1,
                    verbose=0,
                )

                run_name = f"{ablation_slug}__{model_name}__{fs_method}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_param("ablation_name", ablation_name)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("feature_selection", fs_method)
                    mlflow.log_param("test_size", config.test_size)

                    # Search and fit using training data only.
                    grid.fit(X_train, y_train)
                    fitted_pipeline = grid.best_estimator_

                    # Evaluate once on the held-out test data.
                    y_pred = fitted_pipeline.predict(X_test)
                    test_r2 = float(r2_score(y_test, y_pred))
                    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    test_mae = float(mean_absolute_error(y_test, y_pred))
                    cv_best_rmse = float(-grid.best_score_)

                    fs_step = fitted_pipeline.named_steps.get("fs")
                    if fs_step is None:
                        selected_features = feature_columns
                    else:
                        selected_features = np.asarray(feature_columns)[
                            fs_step.get_support()
                        ].tolist()

                    model_path = (
                        results_dir
                        / "artifacts"
                        / ablation_slug
                        / model_name
                        / fs_method
                        / "model.joblib"
                    )
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(fitted_pipeline, model_path)

                    mlflow.log_params(grid.best_params_)
                    mlflow.log_metrics({
                        "cv_best_rmse": cv_best_rmse,
                        "test_r2": test_r2,
                        "test_rmse": test_rmse,
                        "test_mae": test_mae,
                    })
                    mlflow.log_artifact(str(model_path))

                    summary_rows.append({
                        "Ablation": ablation_name,
                        "Model": model_name,
                        "FS Method": fs_method,
                        "Selected Features Count": len(selected_features),
                        "Selected Features": ", ".join(selected_features),
                        "Best Params": str(grid.best_params_),
                        "CV Best RMSE": cv_best_rmse,
                        "Test R2": test_r2,
                        "Test RMSE": test_rmse,
                        "Test MAE": test_mae,
                    })

                    print(
                        f"{run_name}: CV RMSE={cv_best_rmse:.3f}, "
                        f"test R2={test_r2:.3f}, "
                        f"test RMSE={test_rmse:.3f}, "
                        f"test MAE={test_mae:.3f}"
                    )

    if not summary_rows:
        raise RuntimeError("No model combinations were trained.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(
        results_dir / "detailed_regression_results.xlsx",
        index=False,
    )

    # Select the best combination using CV, not the held-out test scores.
    best_row = summary_df.loc[summary_df["CV Best RMSE"].idxmin()]
    best_summary_df = pd.DataFrame([{
        "Best Model": best_row["Model"],
        "Best Ablation": best_row["Ablation"],
        "Best FS Method": best_row["FS Method"],
        "Selected Features Count": best_row["Selected Features Count"],
        "Best Params": best_row["Best Params"],
        "CV Best RMSE": best_row["CV Best RMSE"],
        "Test R2": best_row["Test R2"],
        "Test RMSE": best_row["Test RMSE"],
        "Test MAE": best_row["Test MAE"],
    }])
    best_summary_df.to_excel(
        results_dir / "best_model_summary.xlsx",
        index=False,
    )

    print(f"Training and held-out evaluation completed: {results_dir}")
    return summary_df, best_summary_df

def parse_args():
    parser = argparse.ArgumentParser(description="Train crop height SAR models.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config YAML file")
    parser.add_argument("--retrain", action="store_true", help="Force retraining")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    run_training(cfg, retrain=args.retrain)

if __name__ == "__main__":
    main()