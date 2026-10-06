from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .data import get_ablation_features, load_dataset
from .explain import load_model, plot_saved_model_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Create plots using saved models without retraining."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    results_dir = Path(config.results_dir)
    df = load_dataset(config.data_path)
    indices_path = results_dir / "evaluation" / "test_indices.json"
    if not indices_path.exists():
        raise FileNotFoundError(
            f"Test indices not found: {indices_path}. Run the corrected training command first."
        )

    test_indices = pd.read_json(indices_path, typ="series").to_numpy(dtype=int)
    df_test = df.iloc[test_indices]
    y_test = df_test["Crop_Height"]

    combined_sets = [
        item for item in config.ablation_sets
        if "combined" in item["name"].casefold()
    ]
    if not combined_sets:
        raise ValueError("No ablation set with 'Combined' in its name was found.")

    predictions = []
    missing_models = []

    for ablation_config in combined_sets:
        ablation_name = ablation_config["name"]
        ablation_slug = ablation_name.replace(" ", "_")
        feature_columns = get_ablation_features(df, ablation_config)

        if not feature_columns:
            print(f"Skipping ablation with no features: {ablation_name}")
            continue

        X = df[feature_columns]
        X_test = df_test[feature_columns]

        for model_name in config.models:
            for fs_method in config.feature_selection_methods:
                model_path = (
                    results_dir
                    / "artifacts"
                    / ablation_slug
                    / model_name
                    / fs_method
                    / "model.joblib"
                )

                if not model_path.exists():
                    missing_models.append(str(model_path))
                    continue

                predictions.append({
                    "ablation": ablation_name,
                    "model_name": model_name,
                    "fs_method": fs_method,
                    "model": load_model(model_path),
                    "X": X_test,
                })

    if not predictions:
        raise FileNotFoundError(
            "No saved model.joblib files were found under results/artifacts. "
            "Check the artifact paths; this command does not train models."
        )

    metrics = plot_saved_model_predictions(
        predictions=predictions,
        y=y_test,
        output_dir=results_dir,
    )

    print(f"Created plots and metrics in: {results_dir}")
    print(f"Loaded {len(predictions)} saved models.")
    if missing_models:
        print(f"Skipped {len(missing_models)} missing model files.")


if __name__ == "__main__":
    main()