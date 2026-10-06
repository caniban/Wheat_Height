from __future__ import annotations

import os
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_model(model_path: str | Path):
    """Load a previously saved fitted model without training it."""
    return joblib.load(model_path)

def explain_fitted_pipeline(
    pipeline,
    X: pd.DataFrame,
    output_dir: str | Path,
    title: str,
    top_n: int = 16,
) -> pd.DataFrame:
    """Create SHAP plots and feature-value ranges for a fitted sklearn pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(pipeline, "named_steps") or "model" not in pipeline.named_steps:
        raise ValueError("The saved object must be a fitted pipeline with a 'model' step.")

    # Apply the already-fitted scaler and feature selector; this does not fit them again.
    transformed = pipeline[:-1].transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    if "fs" in pipeline.named_steps:
        selected_mask = pipeline.named_steps["fs"].get_support()
        feature_names = np.asarray(X.columns)[selected_mask].tolist()
    else:
        feature_names = X.columns.tolist()

    X_transformed = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=X.index,
    )

    estimator = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_transformed, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    # SHAP bar plot
    shap.summary_plot(shap_values, X_transformed, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance\n{title}")
    plt.savefig(output_dir / "shap_summary_bar_bestmodel.png", bbox_inches="tight", dpi=300)
    plt.close()

    # SHAP dot plot
    shap.summary_plot(
        shap_values,
        X_transformed,
        show=False,
        cmap=plt.get_cmap("coolwarm"),
    )
    plt.title(f"SHAP Summary Plot\n{title}", pad=20)
    plt.savefig(output_dir / "shap_summary_dot_bestmodel.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Dependence plots for the top features
    importance = np.abs(shap_values).mean(axis=0)
    ordered_indices = np.argsort(importance)[::-1]
    top_indices = ordered_indices[: min(top_n, len(feature_names))]

    n_cols = 4
    n_rows = (len(top_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_rows),
        dpi=300,
        squeeze=False,
    )
    axes = axes.ravel()

    for position, feature_index in enumerate(top_indices):
        ax = axes[position]
        feature = feature_names[feature_index]
        values = X_transformed.iloc[:, feature_index]

        points = ax.scatter(
            values,
            shap_values[:, feature_index],
            c=values,
            cmap="coolwarm",
            alpha=0.75,
            s=14,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"({chr(97 + position)}) {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP Value")
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04, label=feature)

    for position in range(len(top_indices), len(axes)):
        fig.delaxes(axes[position])

    fig.suptitle(f"SHAP Dependence Plots for Top Features\n{title}", y=1.02)
    fig.tight_layout()
    fig.savefig(
        output_dir / "shap_dependence_plots_bestmodel.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Feature-value ranges for samples with the largest positive/negative SHAP values
    range_rows = []
    for feature_index in top_indices:
        feature = feature_names[feature_index]
        feature_values = X_transformed.iloc[:, feature_index]
        feature_shap = shap_values[:, feature_index]

        n_extreme = max(1, int(len(feature_shap) * 0.05))
        positive_indices = np.argsort(feature_shap)[-n_extreme:]
        negative_indices = np.argsort(feature_shap)[:n_extreme]

        range_rows.append({
            "Feature": feature,
            "Positive SHAP Feature Value Min": feature_values.iloc[positive_indices].min(),
            "Positive SHAP Feature Value Max": feature_values.iloc[positive_indices].max(),
            "Negative SHAP Feature Value Min": feature_values.iloc[negative_indices].min(),
            "Negative SHAP Feature Value Max": feature_values.iloc[negative_indices].max(),
        })

    ranges = pd.DataFrame(range_rows)
    ranges.to_excel(output_dir / "shap_feature_value_ranges.xlsx", index=False)

    return ranges

def plot_saved_model_predictions(
    predictions: list[dict],
    y: pd.Series,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Plot predictions from saved fitted models without retraining."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions:
        raise ValueError("No saved model predictions were provided.")

    colors = {
        "RandomForest": "#0072B2",
        "XGBoost": "#D55E00",
        "LightGBM": "#009E73",
    }
    fs_names = {
        "ALL": "Full Dataset",
        "MI": "Selected (Mutual Information)",
        "LASSO": "Selected (Lasso Regression)",
        "RFE": "Selected (RFE)",
    }

    rows = []
    for item in predictions:
        y_pred = np.asarray(item["model"].predict(item["X"]))
        y_true = np.asarray(y)
        rows.append({
            "Ablation": item["ablation"],
            "Model": item["model_name"],
            "FS Method": item["fs_method"],
            "R2 (In-sample)": r2_score(y_true, y_pred),
            "RMSE (In-sample)": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE (In-sample)": mean_absolute_error(y_true, y_pred),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_excel(output_dir / "saved_model_metrics_in_sample.xlsx", index=False)

    n_cols = 4
    n_rows = (len(predictions) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows),
        dpi=300, squeeze=False
    )
    axes = axes.ravel()

    y_true = np.asarray(y)
    for i, item in enumerate(predictions):
        y_pred = np.asarray(item["model"].predict(item["X"]))
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        ax = axes[i]
        ax.scatter(
            y_true, y_pred, alpha=0.6, s=8,
            color=colors.get(item["model_name"], "#8c564b")
        )
        limits = [
            min(float(y_true.min()), float(y_pred.min())),
            max(float(y_true.max()), float(y_pred.max())),
        ]
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set_xlabel("Actual Crop Height (cm)")
        ax.set_ylabel("Predicted Crop Height (cm)")
        ax.set_title(
            f"({chr(97 + i)}) {item['model_name']} - "
            f"{fs_names.get(item['fs_method'], item['fs_method'])}\n"
            f"{item['ablation']}"
        )
        ax.text(
            0.04, 0.96,
            f"In-sample RMSE: {rmse:.2f}\nR²: {r2:.2f}\nMAE: {mae:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
        )

    for i in range(len(predictions), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Actual vs Predicted Crop Height (In-Sample)", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted_saved_models.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows),
        dpi=300, squeeze=False
    )
    axes = axes.ravel()

    all_residuals = [
        np.asarray(y) - np.asarray(item["model"].predict(item["X"]))
        for item in predictions
    ]
    residual_limit = max(float(np.max(np.abs(np.concatenate(all_residuals)))), 1.0)

    for i, (item, residuals) in enumerate(zip(predictions, all_residuals)):
        y_pred = np.asarray(item["model"].predict(item["X"]))
        ax = axes[i]
        ax.scatter(
            y_pred, residuals, alpha=0.6, s=8,
            color=colors.get(item["model_name"], "#8c564b")
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(-residual_limit, residual_limit)
        ax.set_xlabel("Predicted Crop Height (cm)")
        ax.set_ylabel("Residuals (cm)")
        ax.set_title(
            f"({chr(97 + i)}) {item['model_name']} - "
            f"{fs_names.get(item['fs_method'], item['fs_method'])}\n"
            f"{item['ablation']}"
        )
        ax.grid(True, linestyle=":", alpha=0.5)

    for i in range(len(predictions), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Residual Plots for Saved Models (In-Sample)", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "residual_plots_saved_models.png", bbox_inches="tight")
    plt.close(fig)

    return metrics_df