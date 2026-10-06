import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import shap

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, RFE, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
CONFIG = {
    "data_path": "crop_heights.xlsx",
    "random_state": 42,
    "cv_folds": 5,
    "n_features_to_select": 19,
    "mlflow_tracking_uri": "sqlite:///mlruns.db",
    "experiment_name": "Crop_Height_SAR_Estimation",
    "results_dir": os.path.join(os.getcwd(), "results")
}

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
mlflow.set_experiment(CONFIG["experiment_name"])

FS_METHODS = ["ALL", "MI", "LASSO", "RFE"]

MODEL_LIBRARY = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        random_state=CONFIG["random_state"],
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CONFIG["random_state"],
        n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=30,
        boosting_type="gbdt",
        random_state=CONFIG["random_state"],
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

# ---------------------------------------------------------
# 2. DATA LOAD / PREPROCESS
# ---------------------------------------------------------
def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.replace("TSX_", "TDX_")

    if "PointID" in df.columns:
        df[["Year", "Visit", "Point"]] = df["PointID"].str.split("_", expand=True)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    feature_cols = [col for col in df.columns if col.startswith(("PAZ_", "S1", "TDX_"))]
    X = df[feature_cols]
    y = df["Crop_Height"]
    years = df["Year"]

    return X, y, years, feature_cols

# ---------------------------------------------------------
# 3. PIPELINE / FEATURE SELECTION
# ---------------------------------------------------------
def build_pipeline(model, fs_method, n_features):
    steps = [("scaler", StandardScaler())]

    if fs_method == "ALL":
        steps.append(("model", model))
        return Pipeline(steps)

    if fs_method == "MI":
        steps.append(("fs", SelectKBest(score_func=mutual_info_regression, k=n_features)))
    elif fs_method == "LASSO":
        steps.append(("fs", SelectFromModel(
            LassoCV(cv=5, random_state=CONFIG["random_state"]),
            max_features=n_features
        )))
    elif fs_method == "RFE":
        base_estimator = RandomForestRegressor(
            n_estimators=50,
            random_state=CONFIG["random_state"],
            n_jobs=-1
        )
        steps.append(("fs", RFE(estimator=base_estimator, n_features_to_select=n_features)))
    else:
        raise ValueError(f"Unsupported fs_method: {fs_method}")

    steps.append(("model", model))
    return Pipeline(steps)

def get_selected_features(X, y, feature_cols, fs_method):
    if fs_method == "ALL":
        return list(feature_cols)

    n_features = min(len(feature_cols), CONFIG["n_features_to_select"])
    dummy_model = RandomForestRegressor(random_state=CONFIG["random_state"], n_jobs=-1)
    pipe = build_pipeline(dummy_model, fs_method, n_features=n_features)
    pipe.fit(X, y)

    if "fs" not in pipe.named_steps:
        return list(feature_cols)

    selected_mask = pipe.named_steps["fs"].get_support()
    return np.array(feature_cols)[selected_mask].tolist()

def get_ablation_sets(feature_cols):
    c_band_features = [col for col in feature_cols if col.startswith("S1")]
    x_band_features = [col for col in feature_cols if col.startswith(("PAZ_", "TDX_"))]

    return {
        "Only C-Band (Sentinel-1)": c_band_features,
        "Only X-Band (PAZ & TDX)": x_band_features,
        "Combined (C-Band + X-Band)": feature_cols
    }

# ---------------------------------------------------------
# 4. PLOT / ARTIFACT HELPERS
# ---------------------------------------------------------
def save_and_log_correlation(X, filename="correlation_matrix.png"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0, xticklabels=False, yticklabels=False)
    plt.title("SAR Features Correlation Matrix")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(filename)
    os.remove(filename)

def print_feature_selection_summary(X, y, feature_cols, fs_methods):
    rows = []
    for fs_method in fs_methods:
        selected = get_selected_features(X, y, feature_cols, fs_method)
        rows.append({
            "FS Method": fs_method,
            "Number of Features": len(selected),
            "Selected Features": ", ".join(selected)
        })

    fs_summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("--- Feature Selection Summary by Method ---")
    print("=" * 80)
    print(fs_summary_df.to_string(index=False))
    return fs_summary_df

def generate_shap_feature_value_ranges(X_df, shap_values, top_n=16, output_path=None):
    mean_abs_shap = np.abs(shap_values).mean(0)
    feature_importance_order = X_df.columns[np.argsort(mean_abs_shap)[::-1]]
    top_features = feature_importance_order[:top_n]

    shap_analysis_results = []

    for feature in top_features:
        feature_index = X_df.columns.get_loc(feature)
        feature_values = X_df[feature]
        shap_values_feature = shap_values[:, feature_index]

        num_extreme_samples = max(1, int(len(shap_values_feature) * 0.05))
        top_positive_indices = np.argsort(shap_values_feature)[-num_extreme_samples:]
        top_negative_indices = np.argsort(shap_values_feature)[:num_extreme_samples]

        positive_value_range = (np.nan, np.nan)
        if len(top_positive_indices) > 0:
            min_pos = feature_values.iloc[top_positive_indices].min()
            max_pos = feature_values.iloc[top_positive_indices].max()
            positive_value_range = (min_pos, max_pos)

        negative_value_range = (np.nan, np.nan)
        if len(top_negative_indices) > 0:
            min_neg = feature_values.iloc[top_negative_indices].min()
            max_neg = feature_values.iloc[top_negative_indices].max()
            negative_value_range = (min_neg, max_neg)

        shap_analysis_results.append({
            "Feature": feature,
            "Positive SHAP Value Range (Feature Values)": positive_value_range,
            "Negative SHAP Value Range (Feature Values)": negative_value_range
        })

    shap_analysis_df = pd.DataFrame(shap_analysis_results)
    if output_path is not None:
        shap_analysis_df.to_excel(output_path, index=False)

    return shap_analysis_df

# ---------------------------------------------------------
# 5. DETAILED EVALUATION + MLflow LOGGING
# ---------------------------------------------------------
def run_experiments(X, y, years, feature_cols):
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    fs_methods = FS_METHODS
    kf = KFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["random_state"])
    ablation_sets = get_ablation_sets(feature_cols)

    print_feature_selection_summary(X, y, feature_cols, fs_methods)

    all_summary_rows = []
    all_fold_rows = []

    with mlflow.start_run(run_name="Full_SAR_Evaluation") as parent_run:
        mlflow.log_params(CONFIG)
        mlflow.log_param("dataset_rows", len(X))
        mlflow.log_param("dataset_cols", len(feature_cols))
        mlflow.log_param("feature_count", len(feature_cols))
        save_and_log_correlation(X)

        print("\n" + "=" * 80)
        print("--- Detailed Regression Results ---")
        print("=" * 80)

        for ablation_name, ablation_features in ablation_sets.items():
            X_ablation = X[ablation_features]
            print(f"\n--- Ablation Set: {ablation_name} ---")

            for model_name, base_model in MODEL_LIBRARY.items():
                for fs_method in fs_methods:
                    run_name = f"{ablation_name}__{model_name}__{fs_method}"

                    with mlflow.start_run(run_name=run_name, nested=True):
                        mlflow.log_param("ablation_name", ablation_name)
                        mlflow.log_param("ablation_group", ablation_name)
                        mlflow.log_param("model_type", model_name)
                        mlflow.log_param("feature_selection", fs_method)
                        mlflow.log_param("feature_count_in_ablation", len(ablation_features))
                        mlflow.log_param("n_features_to_select", min(len(ablation_features), CONFIG["n_features_to_select"]))

                        pipe = build_pipeline(
                            clone(base_model),
                            fs_method,
                            n_features=min(len(ablation_features), CONFIG["n_features_to_select"])
                        )

                        param_grid = {f"model__{k}": v for k, v in PARAM_GRIDS[model_name].items()}
                        grid = GridSearchCV(
                            estimator=pipe,
                            param_grid=param_grid,
                            scoring="neg_root_mean_squared_error",
                            cv=KFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["random_state"]),
                            refit=True,
                            n_jobs=-1,
                            verbose=0
                        )

                        grid.fit(X_ablation, y)
                        tuned_model = grid.best_estimator_

                        mlflow.log_params(grid.best_params_)
                        mlflow.log_param("best_model_params", str(grid.best_params_))

                        cv_results = cross_validate(
                            tuned_model,
                            X_ablation,
                            y,
                            cv=kf,
                            scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
                            return_estimator=False,
                            n_jobs=-1
                        )

                        mean_r2 = float(np.mean(cv_results["test_r2"]))
                        mean_rmse = float(-np.mean(cv_results["test_neg_root_mean_squared_error"]))
                        mean_mae = float(-np.mean(cv_results["test_neg_mean_absolute_error"]))

                        print(f"\n  Model: {model_name}, FS Method: {fs_method}")
                        print(f"  Best parameters: {grid.best_params_}")
                        print(f"  Average | R2: {mean_r2:.3f} | RMSE: {mean_rmse:.2f} cm | MAE: {mean_mae:.2f} cm")

                        for fold_idx in range(kf.get_n_splits()):
                            r2 = cv_results["test_r2"][fold_idx]
                            rmse = -cv_results["test_neg_root_mean_squared_error"][fold_idx]
                            mae = -cv_results["test_neg_mean_absolute_error"][fold_idx]
                            print(f"    Fold {fold_idx + 1} | R2: {r2:.3f} | RMSE: {rmse:.2f} cm | MAE: {mae:.2f} cm")

                            all_fold_rows.append({
                                "Ablation": ablation_name,
                                "Model": model_name,
                                "FS Method": fs_method,
                                "Fold": fold_idx + 1,
                                "R2": r2,
                                "RMSE": rmse,
                                "MAE": mae,
                                "Best Params": str(grid.best_params_)
                            })

                        selected_features = get_selected_features(X_ablation, y, ablation_features, fs_method)
                        selected_features_count = len(selected_features)

                        all_summary_rows.append({
                            "Ablation": ablation_name,
                            "Model": model_name,
                            "FS Method": fs_method,
                            "Selected Features Count": selected_features_count,
                            "Selected Features": ", ".join(selected_features),
                            "Best Params": str(grid.best_params_),
                            "Mean R2": mean_r2,
                            "Mean RMSE": mean_rmse,
                            "Mean MAE": mean_mae
                        })

                        mlflow.log_metrics({
                            "cv_mean_r2": mean_r2,
                            "cv_mean_rmse": mean_rmse,
                            "cv_mean_mae": mean_mae
                        })

                        mlflow.sklearn.log_model(
                            sk_model=tuned_model,
                            artifact_path=f"model_{model_name}_{fs_method}_{ablation_name.replace(' ', '_')}",
                            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                        )

    summary_df = pd.DataFrame(all_summary_rows)
    fold_df = pd.DataFrame(all_fold_rows)

    summary_path = os.path.join(CONFIG["results_dir"], "detailed_regression_results.xlsx")
    fold_path = os.path.join(CONFIG["results_dir"], "detailed_regression_fold_results.xlsx")
    summary_df.to_excel(summary_path, index=False)
    fold_df.to_excel(fold_path, index=False)

    print("\n" + "=" * 80)
    print("--- Tuned Model Hyperparameters ---")
    print("=" * 80)
    print(summary_df[["Ablation", "Model", "FS Method", "Selected Features Count", "Best Params"]].to_string(index=False))

    best_row = summary_df.sort_values("Mean R2", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_ablation_name = best_row["Ablation"]
    best_fs_method = best_row["FS Method"]

    best_summary_df = pd.DataFrame([{
        "Best Model": best_model_name,
        "Best Ablation": best_ablation_name,
        "Best FS Method": best_fs_method,
        "Selected Features Count": best_row["Selected Features Count"],
        "Best Params": best_row["Best Params"],
        "Mean R2": best_row["Mean R2"],
        "Mean RMSE": best_row["Mean RMSE"],
        "Mean MAE": best_row["Mean MAE"]
    }])
    best_summary_path = os.path.join(CONFIG["results_dir"], "best_model_summary.xlsx")
    best_summary_df.to_excel(best_summary_path, index=False)

    print("\nBest combination:")
    print(best_summary_df.to_string(index=False))

    best_ablation_features = get_ablation_sets(feature_cols)[best_ablation_name]
    X_best = X[best_ablation_features]

    best_pipe = build_pipeline(
        clone(MODEL_LIBRARY[best_model_name]),
        best_fs_method,
        n_features=min(len(best_ablation_features), CONFIG["n_features_to_select"])
    )

    grid = GridSearchCV(
        estimator=best_pipe,
        param_grid={f"model__{k}": v for k, v in PARAM_GRIDS[best_model_name].items()},
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["random_state"]),
        refit=True,
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_best, y)
    best_model = grid.best_estimator_

    plot_actual_vs_predicted(X, y, MODEL_LIBRARY, FS_METHODS, CONFIG["results_dir"])
    plot_residuals(X, y, MODEL_LIBRARY, FS_METHODS, CONFIG["results_dir"])
    log_shap_analysis(best_model, X_best, y, best_ablation_features, title=f"{best_model_name} - {best_ablation_name} - {best_fs_method}")

    return summary_df, fold_df, best_summary_df

# ---------------------------------------------------------
# 6. Actual vs predicted plots
# ---------------------------------------------------------
def plot_actual_vs_predicted(X, y, models, fs_methods, output_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=CONFIG["random_state"]
    )

    plot_combinations = []
    for model_name in models.keys():
        for fs_method in fs_methods:
            plot_combinations.append((model_name, fs_method))

    n_models_to_plot = len(plot_combinations)
    n_cols = 4
    n_rows = (n_models_to_plot + n_cols - 1) // n_cols

    subplot_size = 3.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(subplot_size * n_cols, subplot_size * n_rows), dpi=300)
    axes = axes.flatten()

    regressor_colors = {
        "RandomForest": "#0072B2",
        "XGBoost": "#D55E00",
        "LightGBM": "#009E73"
    }

    feature_set_display_names = {
        "ALL": "Full Dataset",
        "MI": "Selected (Mutual Information)",
        "LASSO": "Selected (Lasso Regression)",
        "RFE": "Selected (RFE)"
    }

    for i, (model_name, fs_method) in enumerate(plot_combinations):
        pipe = build_pipeline(
            clone(models[model_name]),
            fs_method,
            n_features=min(len(X.columns), CONFIG["n_features_to_select"])
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.6, s=8, color=regressor_colors.get(model_name, "#8c564b"))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=1)

        ax.set_xlabel("Actual Crop Height (cm)", fontsize=8)
        ax.set_ylabel("Predicted Crop Height (cm)", fontsize=8)
        ax.set_title(f"({chr(97 + i)}) {model_name}\n{feature_set_display_names[fs_method]}", fontsize=10)
        ax.text(
            0.05, 0.95,
            f"RMSE: {rmse:.2f} cm\nR²: {r2:.2f}\nMAE: {mae:.2f} cm",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", linewidth=0.5)
        )
        ax.tick_params(axis="both", which="major", labelsize=7)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=1.5)
    plt.suptitle("Actual vs Predicted Crop Height for Evaluated Models", fontsize=14, y=1.02)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "actual_vs_predicted_all_models_fs.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------
# 7. Residual plots
# ---------------------------------------------------------
def plot_residuals(X, y, models, fs_methods, output_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=CONFIG["random_state"]
    )

    all_predictions = {}
    all_residuals = {}
    for model_name, model in models.items():
        for fs_method in fs_methods:
            combo_name = f"{model_name}_{fs_method}"
            pipe = build_pipeline(
                clone(model),
                fs_method,
                n_features=min(len(X.columns), CONFIG["n_features_to_select"])
            )
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            residuals = y_test - pred
            all_predictions[combo_name] = pred
            all_residuals[combo_name] = residuals

    n_models_to_plot = len(all_predictions)
    n_cols = 4
    n_rows = (n_models_to_plot + n_cols - 1) // n_cols
    subplot_size = 3.5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(subplot_size * n_cols, subplot_size * n_rows), dpi=300)
    axes = axes.flatten()

    regressor_colors = {
        "RandomForest": "#0072B2",
        "XGBoost": "#D55E00",
        "LightGBM": "#009E73"
    }

    feature_set_display_names = {
        "ALL": "Full Dataset",
        "MI": "Selected (Mutual Information)",
        "LASSO": "Selected (Lasso Regression)",
        "RFE": "Selected (RFE)"
    }

    all_res_values = np.concatenate(list(all_residuals.values()))
    y_min = np.min(all_res_values) - 5
    y_max = np.max(all_res_values) + 5
    y_ticks = np.linspace(y_min, y_max, num=5)

    for i, (combo_name, residuals) in enumerate(all_residuals.items()):
        ax = axes[i]
        predicted_values = all_predictions[combo_name]
        model_name, fs_method = combo_name.split("_", 1)

        ax.scatter(predicted_values, residuals, alpha=0.6, s=5, color=regressor_colors.get(model_name, "#999999"))
        ax.axhline(y=0, color="black", linestyle="--", lw=1)

        ax.set_xlabel("Predicted Crop Height (cm)", fontsize=8)
        ax.set_ylabel("Residuals (cm)", fontsize=8)
        ax.set_ylim([y_min, y_max])
        ax.set_yticks(y_ticks)
        ax.set_title(f"({chr(97 + i)}) {model_name}\n{feature_set_display_names[fs_method]}", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.grid(True, linestyle=":", alpha=0.6, color="gray")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=1.5)
    plt.suptitle("Residual Plots for Evaluated Models", fontsize=14, y=1.03)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "residual_plots_all_models_fs.png"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------
# 8. SHAP analysis
# ---------------------------------------------------------
def log_shap_analysis(fitted_pipeline, X, y, feature_cols, title="Best Model"):
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=CONFIG["random_state"]
    )

    scaler = fitted_pipeline.named_steps["scaler"]
    model_step = fitted_pipeline.named_steps["model"]

    if "fs" in fitted_pipeline.named_steps:
        fs_step = fitted_pipeline.named_steps["fs"]
        selected_mask = fs_step.get_support()
        selected_feature_names = np.array(feature_cols)[selected_mask]
        X_test_selected = fs_step.transform(scaler.transform(X_test))
        X_test_df = pd.DataFrame(X_test_selected, columns=selected_feature_names)
    else:
        X_test_df = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    explainer = shap.TreeExplainer(model_step)
    shap_values = explainer.shap_values(X_test_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # SHAP BAR PLOT
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title(f"Feature Importance (SHAP values)\n({title})", fontsize=16)
    path_bar = os.path.join(CONFIG["results_dir"], "shap_summary_bar_bestmodel.png")
    plt.savefig(path_bar, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(path_bar)

    # SHAP DOT PLOT
    plt.figure(figsize=(10, 8), dpi=300)
    cmap = plt.get_cmap("coolwarm")
    shap.summary_plot(shap_values, X_test_df, show=False, cmap=cmap)
    plt.title(f"SHAP Summary Plot for Model Explainability\n({title})", fontsize=16, pad=20)
    path_dot = os.path.join(CONFIG["results_dir"], "shap_summary_dot_bestmodel.png")
    plt.savefig(path_dot, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(path_dot)

    # SHAP dependence plots (top 16)
    mean_abs_shap = np.abs(shap_values).mean(0)
    feature_importance_order = X_test_df.columns[np.argsort(mean_abs_shap)[::-1]]
    top_features = feature_importance_order[:16]

    n_rows = 4
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows), dpi=300)
    axes = axes.flatten()

    subplot_labels = [chr(ord("a") + i) for i in range(len(top_features))]

    for i, feature in enumerate(top_features):
        ax = axes[i]

        temp_fig, temp_ax = plt.subplots()
        shap.dependence_plot(
            feature,
            shap_values,
            X_test_df,
            interaction_index=None,
            show=False,
            ax=temp_ax
        )

        if temp_ax.collections:
            collection = temp_ax.collections[0]
            offsets = collection.get_offsets()
            colors = collection.get_array()
            sizes = collection.get_sizes()
            cmap = collection.get_cmap()

            ax.scatter(offsets[:, 0], offsets[:, 1], c=colors, s=sizes, cmap=cmap)
            ax.set_xlim(temp_ax.get_xlim())
            ax.set_ylim(temp_ax.get_ylim())
            ax.set_xticks(temp_ax.get_xticks())
            ax.set_yticks(temp_ax.get_yticks())

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"({subplot_labels[i]}) {feature}", fontsize=12)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("SHAP Value", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, linestyle=":", alpha=0.6, color="gray")
        plt.close(temp_fig)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0)
    plt.suptitle("SHAP Dependency Plots for Top Features", fontsize=16, y=1.02)
    dependence_path = os.path.join(CONFIG["results_dir"], "shap_dependence_plots_bestmodel.png")
    fig.savefig(dependence_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(dependence_path)

    shap_ranges_df = generate_shap_feature_value_ranges(
        X_test_df,
        shap_values,
        top_n=16,
        output_path=os.path.join(CONFIG["results_dir"], "shap_feature_value_ranges.xlsx")
    )
    mlflow.log_artifact(os.path.join(CONFIG["results_dir"], "shap_feature_value_ranges.xlsx"))

    print(f"\nSHAP artifacts saved to: {CONFIG['results_dir']}")
    print(f"  - SHAP Bar Plot: {path_bar}")
    print(f"  - SHAP Dot Plot: {path_dot}")
    print(f"  - SHAP Dependence Plot: {dependence_path}")
    print(f"  - SHAP Feature Value Ranges: {os.path.join(CONFIG['results_dir'], 'shap_feature_value_ranges.xlsx')}")

# ---------------------------------------------------------
# 9. Main execution
# ---------------------------------------------------------
if __name__ == "__main__":
    X, y, years, feature_cols = load_and_preprocess_data(CONFIG["data_path"])
    summary_df, fold_df, best_summary_df = run_experiments(X, y, years, feature_cols)

    print("\nAll outputs saved successfully.")
    print(f"Result folder: {CONFIG['results_dir']}")
    print("MLflow UI command: mlflow ui --backend-store-uri sqlite:///mlruns.db")