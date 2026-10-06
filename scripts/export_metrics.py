from pathlib import Path

import mlflow
import pandas as pd

TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "Crop_Height_SAR_Estimation"
OUTPUT_PATH = Path("results") / "all_model_metrics.xlsx"

mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise SystemExit(
        f"Experiment not found: {EXPERIMENT_NAME}. "
        "Check the tracking URI and run this script from the project root."
    )

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    output_format="pandas",
)

if runs.empty:
    raise SystemExit("No MLflow runs found for this experiment.")

metric_columns = [column for column in runs.columns if column.startswith("metrics.")]
if not metric_columns:
    raise SystemExit("The experiment contains no logged metrics.")

runs = runs.loc[runs[metric_columns].notna().any(axis=1)].copy()

if runs.empty:
    raise SystemExit("No runs with logged metrics were found.")

runs.insert(
    0,
    "run_name",
    runs.get("tags.mlflow.runName", pd.Series(index=runs.index, dtype="object")),
)

preferred_columns = [
    "run_name",
    "params.ablation_name",
    "params.model_type",
    "params.feature_selection",
    "metrics.cv_mean_r2",
    "metrics.cv_mean_rmse",
    "metrics.cv_mean_mae",
    "status",
    "run_id",
]
columns = [column for column in preferred_columns if column in runs.columns]
columns += [
    column
    for column in runs.columns
    if column.startswith(("params.", "metrics.")) and column not in columns
]

results = runs[columns].sort_values("run_name", na_position="last")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_excel(OUTPUT_PATH, index=False)

print(f"Exported {len(results)} runs to: {OUTPUT_PATH}")