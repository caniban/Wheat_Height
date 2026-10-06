from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
import yaml
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from cropheight.data import get_ablation_features, load_dataset  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
RESULTS_DIR = ROOT / CONFIG.get("results_dir", "results")
MODEL_DIR = RESULTS_DIR / "artifacts"

st.set_page_config(page_title="Wheat Height | Model Laboratory", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#172B2A; --muted:#60736D; --paper:#F4F6F0; --line:#DCE4D9; --green:#147D64; --orange:#D77932; }
    html, body, [class*="css"] { font-family:'Manrope',sans-serif; color:var(--ink); }
    .stApp { background:var(--paper); }
    [data-testid="stHeader"] { background:rgba(244,246,240,.88); }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:6px; padding:14px 16px; }
    [data-testid="stMetricLabel"] { color:var(--muted); font-size:.78rem; }
    [data-testid="stMetricValue"] { color:var(--ink); font-weight:800; }
    .hero { padding:24px 0 18px; border-bottom:1px solid var(--line); margin-bottom:18px; }
    .hero-kicker { font:500 11px 'DM Mono',monospace; color:var(--green); text-transform:uppercase; }
    .hero h1 { font-size:32px; line-height:1.15; margin:7px 0 5px; }
    .hero p { color:var(--muted); margin:0; font-size:14px; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:5px; }
        div[data-testid="stTabs"] [data-testid="stTab"] {
            font-weight:800;
            border:1px solid #D5DDD7;
            border-bottom:4px solid transparent;
            border-radius:5px 5px 0 0;
            margin-right:5px;
            padding:10px 14px;
            transition:background-color .16s ease,color .16s ease,border-color .16s ease;
        }
        div[data-testid="stTabs"] [data-testid="stTab"][data-key="0"] { background:#D5F1E5; color:#12664F; border-bottom-color:#168365; }
        div[data-testid="stTabs"] [data-testid="stTab"][data-key="1"] { background:#FFE4C7; color:#8D4312; border-bottom-color:#E4772F; }
        div[data-testid="stTabs"] [data-testid="stTab"][data-key="2"] { background:#DCE8FC; color:#315D9B; border-bottom-color:#477AC3; }
        div[data-testid="stTabs"] [data-testid="stTab"][data-key="3"] { background:#FADBD7; color:#A6423B; border-bottom-color:#D45B50; }
        div[data-testid="stTabs"] [data-testid="stTab"][data-key="4"] { background:#EADDF8; color:#6C4A91; border-bottom-color:#8A60B5; }
        div[data-testid="stTabs"] [data-testid="stTab"]:hover { filter:brightness(.96); }
        div[data-testid="stTabs"] [data-testid="stTab"][data-key="0"][aria-selected="true"] { background:#147D64; color:#fff; border-color:#147D64; }
        div[data-testid="stTabs"] [data-testid="stTab"][data-key="1"][aria-selected="true"] { background:#D77932; color:#fff; border-color:#D77932; }
        div[data-testid="stTabs"] [data-testid="stTab"][data-key="2"][aria-selected="true"] { background:#4276A8; color:#fff; border-color:#4276A8; }
        div[data-testid="stTabs"] [data-testid="stTab"][data-key="3"][aria-selected="true"] { background:#C84C4C; color:#fff; border-color:#C84C4C; }
        div[data-testid="stTabs"] [data-testid="stTab"][data-key="4"][aria-selected="true"] { background:#7652A3; color:#fff; border-color:#7652A3; }
    .note { border-left:3px solid var(--orange); background:#fff; padding:12px 15px; color:var(--muted); font-size:13px; }
    </style>
    <div class="hero">
      <div class="hero-kicker">Crop-height estimation | SAR regression</div>
      <h1>Wheat Height Model Laboratory</h1>
      <p>Experiment design, fitted models, held-out evaluation, and model explanations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def latest_runs() -> pd.DataFrame:
    mlflow.set_tracking_uri(CONFIG.get("mlflow_tracking_uri", "sqlite:///mlruns.db"))
    experiment = mlflow.get_experiment_by_name(CONFIG["experiment_name"])
    if experiment is None:
        return pd.DataFrame()
    runs = mlflow.search_runs([experiment.experiment_id], output_format="pandas")
    if runs.empty or "status" not in runs:
        return pd.DataFrame()
    metrics = [name for name in runs if name.startswith("metrics.")]
    runs = runs.loc[runs.status.eq("FINISHED")].copy()
    if metrics:
        runs = runs.loc[runs[metrics].notna().any(axis=1)].copy()
    keys = ["params.ablation_name", "params.model_name", "params.feature_selection"]
    if any(name not in runs for name in keys):
        return pd.DataFrame()
    ablations = {item["name"] for item in CONFIG["ablation_sets"]}
    runs = runs.loc[
        runs[keys[0]].isin(ablations)
        & runs[keys[1]].isin(CONFIG["models"])
        & runs[keys[2]].isin(CONFIG["feature_selection_methods"])
    ].sort_values("start_time", ascending=False)
    return runs.drop_duplicates(keys, keep="first").sort_values(keys)


@st.cache_data(show_spinner=False)
def observations() -> pd.DataFrame:
    return load_dataset(CONFIG.get("data_path", "data/crop_heights.xlsx"))


@st.cache_data(show_spinner=False)
def vif_table(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.median())
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.empty:
        return pd.DataFrame(columns=["Feature", "VIF", "Tolerance", "Status"])
    design = add_constant(numeric, has_constant="add").to_numpy(dtype=float)
    values = [variance_inflation_factor(design, i) for i in range(1, design.shape[1])]
    result = pd.DataFrame({"Feature": numeric.columns, "VIF": values})
    result["Tolerance"] = 1 / result["VIF"].replace(0, np.nan)
    result["Status"] = np.select(
        [result.VIF > 10, result.VIF > 5], ["High", "Moderate"], default="Low"
    )
    return result.sort_values("VIF", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def post_selection_diagnostics(
    runs: pd.DataFrame, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows = []
    for _, run in runs.iterrows():
        path = model_path(run)
        if not path.exists():
            continue
        feature_columns = get_ablation_features(
            data,
            next(item for item in CONFIG["ablation_sets"] if item["name"] == run["params.ablation_name"]),
        )
        pipeline = joblib.load(path)
        selector = pipeline.named_steps.get("fs")
        selected = (
            feature_columns
            if selector is None
            else np.asarray(feature_columns)[selector.get_support()].tolist()
        )
        if not selected:
            continue

        baseline = data[feature_columns].apply(pd.to_numeric, errors="coerce")
        baseline = baseline.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        baseline = baseline.fillna(baseline.median())
        selected_frame = data[selected].apply(pd.to_numeric, errors="coerce")
        selected_frame = selected_frame.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        selected_frame = selected_frame.fillna(selected_frame.median())
        before_corr = baseline.corr().abs().to_numpy()
        before_upper = before_corr[np.triu_indices_from(before_corr, k=1)]
        before_vif = vif_table(baseline)
        correlation = selected_frame.corr().abs().to_numpy()
        upper = correlation[np.triu_indices_from(correlation, k=1)]
        diagnostics = vif_table(selected_frame)
        before_corr_pairs = int((before_upper >= 0.8).sum())
        after_corr_pairs = int((upper >= 0.8).sum())
        before_vif_count = int(before_vif["VIF"].gt(5).sum())
        after_vif_count = int(diagnostics["VIF"].gt(5).sum())
        if after_corr_pairs == 0 and after_vif_count == 0:
            status = "Resolved (|r| < 0.8 and all VIF <= 5)"
        elif after_corr_pairs > before_corr_pairs or after_vif_count > before_vif_count:
            status = "Mixed/increased; issue remains"
        elif after_corr_pairs < before_corr_pairs or after_vif_count < before_vif_count:
            status = "Reduced; residual issue remains"
        else:
            status = "Persists; no threshold-count reduction"
        identity = {
            "Ablation": run["params.ablation_name"],
            "Model": run["params.model_name"],
            "Feature selection": run["params.feature_selection"],
        }
        summary_rows.append(
            {
                **identity,
                "Selected feature count": len(selected),
                "Selected features": ", ".join(selected),
                "Before max |r|": float(before_upper.max()) if len(before_upper) else np.nan,
                "Before pairs |r| >= 0.8": before_corr_pairs,
                "Before max VIF": float(before_vif["VIF"].max()) if not before_vif.empty else np.nan,
                "Before features VIF > 5": before_vif_count,
                "Maximum |Pearson r|": float(upper.max()) if len(upper) else np.nan,
                "Pairs with |r| >= 0.8": int((upper >= 0.8).sum()),
                "Maximum VIF": float(diagnostics["VIF"].max()) if not diagnostics.empty else np.nan,
                "Features with VIF > 5": int(diagnostics["VIF"].gt(5).sum()),
                "Features with VIF > 10": int(diagnostics["VIF"].gt(10).sum()),
                "Collinearity outcome": status,
            }
        )
        if not diagnostics.empty:
            detail_rows.append(diagnostics.assign(**identity))

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return summary, details


def model_path(run: pd.Series) -> Path:
    ablation = str(run["params.ablation_name"]).replace(" ", "_")
    return (
        MODEL_DIR
        / ablation
        / str(run["params.model_name"])
        / str(run["params.feature_selection"])
        / "model.joblib"
    )


@st.cache_data(show_spinner=False)
def shap_data(
    path: str, columns: tuple[str, ...], test_indices: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = joblib.load(path)
    frame = observations().iloc[list(test_indices)][list(columns)]
    transformed = pipeline[:-1].transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    selector = pipeline.named_steps.get("fs")
    names = list(columns) if selector is None else np.asarray(columns)[selector.get_support()].tolist()
    transformed = pd.DataFrame(transformed, columns=names, index=frame.index)
    values = shap.TreeExplainer(pipeline.named_steps["model"]).shap_values(
        transformed, check_additivity=False
    )
    if isinstance(values, list):
        values = values[0]
    if hasattr(values, "values"):
        values = values.values
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 0]
    return transformed, pd.DataFrame(values, columns=names, index=frame.index)


def workflow_tab(runs: pd.DataFrame, data: pd.DataFrame) -> None:
    st.subheader("Study workflow")
    st.caption("Feature selection and hyperparameter tuning use training data only; the held-out test set is evaluated once.")
    st.graphviz_chart(
        f'''
        digraph study {{
          graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.28", ranksep="0.45"];
          node [shape=box, style="rounded,filled", color="#DCE4D9", fillcolor="#FFFFFF", fontname="Manrope", fontsize=10, margin="0.14,0.10"];
          edge [color="#82968D", penwidth=1.2, arrowsize=0.7];
          source [label="{len(data)} field observations\\nSAR predictors + Crop_Height (cm)", fillcolor="#E5F2EB", color="#147D64"];
          prep [label="Preparation\\nTSX_ -> TDX_; parse PointID"];
          groups [label="Feature groups\\nC-band: S1*\\nX-band: PAZ_* + TDX_*"];
          split [label="Random train / test split\\n{1 - CONFIG.get('test_size', 0.3):.0%} / {CONFIG.get('test_size', 0.3):.0%}; seed 42", fillcolor="#FFF1E5", color="#D77932"];
          ablation [label="Ablations\\nC-band | X-band | combined"];
          diag [label="Before-selection diagnostics\\nAbsolute Pearson r\\nVIF + tolerance"];
          cv [label="Training-only GridSearchCV\\n5-fold shuffled KFold\\nScoring: negative RMSE"];
          pipe [label="Per-fold pipeline\\nStandardScaler -> selector -> model\\nALL | MI | LASSO | SHAP_RFE"];
          models [label="Regressors\\nRandomForest | XGBoost | LightGBM\\nBest pipeline refit on training data"];
          selected [label="Post-selection diagnostics\\nSelected features + correlation\\nVIF + tolerance"];
          test [label="Held-out test evaluation\\nR2 | RMSE | MAE\\nCV best RMSE retained"];
          log [label="MLflow + joblib artifacts\\nParameters | metrics | fitted pipeline"];
          explain [label="Best-CV model explanation\\nTreeSHAP on test observations\\nImportance + feature effects"];
          source -> prep -> groups -> split -> ablation;
          ablation -> diag;
          ablation -> cv -> pipe -> models -> selected -> test -> log -> explain;
        }}
        ''',
        width="stretch",
    )

    st.markdown("### Experimental design")
    metrics = st.columns(4)
    metrics[0].metric("Observations", f"{len(data):,}")
    metrics[1].metric("Target", "Crop_Height")
    metrics[2].metric("Configured models", len(CONFIG["models"]))
    metrics[3].metric("Latest model runs", len(runs))
    st.markdown(
        """
        - **Outcome:** `Crop_Height`, treated as a continuous regression target.
        - **Predictors:** source columns with `S1`, `PAZ_`, and `TSX_` prefixes; the loader normalizes `TSX_` to `TDX_`. `PointID` is parsed into year, visit, and point where available; metadata is not included as a predictor.
        - **Ablations:** Sentinel-1 C-band (`S1*`), PAZ/TDX X-band (`PAZ_*`, `TDX_*`), and combined C + X inputs.
        - **Split and tuning:** shuffled random train/test split with seed 42 and configured 30% test fraction; 5-fold shuffled `KFold` on training data. `GridSearchCV` minimizes negative root mean squared error and refits the best pipeline on training data.
        - **Pipeline order:** `StandardScaler` -> feature selector -> regressor. The fitted pipeline is persisted as `model.joblib`.
        - **Selectors:** `ALL` retains all features; `MI` uses `SelectKBest(mutual_info_regression)`; `LASSO` uses `SelectFromModel(LassoCV, cv=5, max_features=19)`; `SHAP_RFE` repeatedly fits a 100-tree Random Forest and removes the least mean-|TreeSHAP|-important feature until 19 remain.
        - **Regressors:** Random Forest, XGBoost, and LightGBM. Candidate grids and selected best parameters are listed in the model hyperparameters tab.
        - **Evaluation:** test data are not used for selection. Each run logs CV best RMSE and held-out R², RMSE, and MAE.
        - **Collinearity:** absolute Pearson correlation matrices and predictor-level VIF/tolerance are reported before feature selection. VIF includes an intercept; tolerance is `1 / VIF`. Bands: low `≤5`, moderate `>5 to ≤10`, high `>10`.
        """
    )

    st.markdown("### Correlation matrices and multicollinearity")
    st.caption("Absolute Pearson correlation and full predictor-level VIF/tolerance tables for each ablation before feature selection.")
    for item in CONFIG["ablation_sets"]:
        columns = get_ablation_features(data, item)
        if not columns:
            continue
        frame = data[columns].apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        frame = frame.fillna(frame.median())
        st.markdown(f"#### {item['name']} | {len(frame.columns)} predictors")
        left, right = st.columns([1.5, 1])
        with left:
            corr = frame.corr().abs()
            fig, ax = plt.subplots(figsize=(max(8, min(15, len(corr.columns) * 0.35)), 7))
            sns.heatmap(corr, cmap="YlGnBu", vmin=0, vmax=1, square=True, linewidths=0.15,
                        cbar_kws={"label": "Absolute Pearson correlation"}, ax=ax)
            ax.set_title("Absolute Pearson correlation matrix", loc="left", weight="bold")
            ax.tick_params(axis="x", labelrotation=90, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        with right:
            diagnostic = vif_table(frame)
            st.markdown("**Variance inflation and tolerance**")
            st.dataframe(diagnostic.style.format({"VIF": "{:.3f}", "Tolerance": "{:.4f}"}),
                         hide_index=True, width="stretch", height=520)
            high = int(diagnostic.Status.eq("High").sum())
            moderate = int(diagnostic.Status.eq("Moderate").sum())
            st.caption(f"High VIF: {high} | Moderate VIF: {moderate} | predictors: {len(diagnostic)}")

    st.markdown("### Feature-selection methods")
    st.dataframe(pd.DataFrame([
        {"Method": "ALL", "Selection rule": "Retain the complete ablation feature set."},
        {"Method": "MI", "Selection rule": "SelectKBest with mutual_info_regression; k = min(19, available features)."},
        {"Method": "LASSO", "Selection rule": "SelectFromModel with 5-fold LassoCV; at most 19 features."},
        {"Method": "SHAP_RFE", "Selection rule": "Repeated 100-tree Random Forest + TreeSHAP ranking; recursively remove the least important feature until 19 remain."},
    ]), hide_index=True, width="stretch")



def feature_selection_tab(runs: pd.DataFrame, data: pd.DataFrame) -> None:
    st.subheader("Feature selection and post-selection collinearity")
    st.caption("Complete results for every latest MI, LASSO, and SHAP-RFE run across all ablations and model families.")
    methods = ["MI", "LASSO", "SHAP_RFE"]
    selected_runs = runs.loc[runs["params.feature_selection"].isin(methods)].copy()
    with st.spinner("Reading fitted selectors and recalculating diagnostics..."):
        results, vif_details = post_selection_diagnostics(selected_runs, data)
    if results.empty:
        st.warning("No fitted selector artifacts were available for the configured runs.")
        return

    st.markdown(
        """
        The feature list below is read from each fitted pipeline. Before/after diagnostics use the same source observations and numeric predictors. A collinearity issue is marked **Resolved** only when the selected set has no pair with `|r| >= 0.8` and every VIF is `<= 5`; reductions with remaining threshold violations are explicitly marked as residual issues.
        """
    )
    status_counts = results["Collinearity outcome"].value_counts()
    resolved = int(results["Collinearity outcome"].str.startswith("Resolved").sum())
    reduced = int(results["Collinearity outcome"].str.startswith("Reduced").sum())
    remaining = len(results) - resolved - reduced
    stats = st.columns(4)
    stats[0].metric("Selector runs", len(results))
    stats[1].metric("Resolved by thresholds", resolved)
    stats[2].metric("Reduced, residual remains", reduced)
    stats[3].metric("Persists or mixed", remaining)

    display_columns = [
        "Ablation", "Model", "Feature selection", "Selected feature count", "Selected features",
        "Before max |r|", "Maximum |Pearson r|", "Before pairs |r| >= 0.8", "Pairs with |r| >= 0.8",
        "Before max VIF", "Maximum VIF", "Before features VIF > 5", "Features with VIF > 5",
        "Features with VIF > 10", "Collinearity outcome",
    ]
    results = results.sort_values(["Ablation", "Feature selection", "Model"])
    st.markdown("### Selected factors and before/after diagnostics")
    st.dataframe(
        results[display_columns].style.format(
            {"Before max |r|": "{:.3f}", "Maximum |Pearson r|": "{:.3f}",
             "Before max VIF": "{:.3f}", "Maximum VIF": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
        height=520,
    )
    st.download_button(
        "Download all selector results",
        results[display_columns].to_csv(index=False).encode("utf-8"),
        "feature_selection_and_collinearity.csv",
        "text/csv",
    )

    st.markdown("### Correlation matrices after feature selection")
    unique_sets = results.drop_duplicates(["Ablation", "Feature selection", "Selected features"])
    high_pairs = []
    for _, row in unique_sets.iterrows():
        features = row["Selected features"].split(", ")
        matrix = data[features].apply(pd.to_numeric, errors="coerce").corr().abs()
        model_names = ", ".join(
            results.loc[
                results["Ablation"].eq(row["Ablation"])
                & results["Feature selection"].eq(row["Feature selection"])
                & results["Selected features"].eq(row["Selected features"]),
                "Model",
            ].tolist()
        )
        for i, feature_a in enumerate(features):
            for feature_b in features[i + 1:]:
                value = matrix.loc[feature_a, feature_b]
                if pd.notna(value) and value >= 0.8:
                    high_pairs.append({
                        "Ablation": row["Ablation"],
                        "Feature selection": row["Feature selection"],
                        "Models": model_names,
                        "Feature A": feature_a,
                        "Feature B": feature_b,
                        "Absolute Pearson r": value,
                    })

    for start in range(0, len(unique_sets), 2):
        chart_columns = st.columns(2)
        for slot, (_, row) in enumerate(unique_sets.iloc[start:start + 2].iterrows()):
            features = row["Selected features"].split(", ")
            matrix = data[features].apply(pd.to_numeric, errors="coerce").corr().abs()
            model_names = ", ".join(
                results.loc[
                    results["Ablation"].eq(row["Ablation"])
                    & results["Feature selection"].eq(row["Feature selection"])
                    & results["Selected features"].eq(row["Selected features"]),
                    "Model",
                ].tolist()
            )
            with chart_columns[slot]:
                st.markdown(f"**{row['Ablation']} | {row['Feature selection']}**")
                st.caption(f"Selected features: {len(features)} | Models: {model_names}")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(matrix, cmap="YlGnBu", vmin=0, vmax=1, square=True,
                            linewidths=0.15, cbar_kws={"label": "Absolute Pearson correlation"}, ax=ax)
                ax.tick_params(axis="x", labelrotation=90, labelsize=6)
                ax.tick_params(axis="y", labelsize=6)
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

    st.markdown("### Remaining high-correlation pairs")
    st.caption("All selected-feature pairs with absolute Pearson correlation >= 0.8; an empty table means no selected set retained such a pair.")
    if high_pairs:
        st.dataframe(
            pd.DataFrame(high_pairs).style.format({"Absolute Pearson r": "{:.3f}"}),
            hide_index=True,
            width="stretch",
            height=360,
        )
    else:
        st.success("No selected feature set contains a pair with |r| >= 0.8.")

    st.markdown("### Feature-level VIF and tolerance after selection")
    st.caption("One row per selected feature and fitted run, with its VIF, tolerance, and severity band.")
    st.dataframe(
        vif_details[["Ablation", "Model", "Feature selection", "Feature", "VIF", "Tolerance", "Status"]]
        .style.format({"VIF": "{:.3f}", "Tolerance": "{:.4f}"}),
        hide_index=True,
        width="stretch",
        height=520,
    )


def hyperparameters_tab(runs: pd.DataFrame) -> None:
    st.subheader("Model hyperparameters")
    st.caption("All MLflow parameters for the latest successful run of every configured ablation, model, and feature-selection combination.")
    identity = ["tags.mlflow.runName", "params.ablation_name", "params.model_name",
                "params.feature_selection", "start_time", "run_id"]
    params = sorted(name for name in runs if name.startswith("params."))
    columns = [name for name in identity if name in runs] + [name for name in params if name not in identity]
    table = runs[columns].rename(columns=lambda name: name.removeprefix("params.").removeprefix("tags.mlflow."))
    st.dataframe(table, hide_index=True, width="stretch", height=680)
    st.download_button("Download complete parameter table", table.to_csv(index=False).encode("utf-8"),
                       "latest_model_hyperparameters.csv", "text/csv")


def metrics_tab(runs: pd.DataFrame) -> None:
    st.subheader("Held-out and cross-validation metrics")
    st.caption("Complete metric table from MLflow. Every metric column is shown; blank cells mean that a metric was not logged for that run.")
    identity = ["tags.mlflow.runName", "params.ablation_name", "params.model_name",
                "params.feature_selection", "start_time", "run_id", "status"]
    metrics = sorted(name for name in runs if name.startswith("metrics."))
    columns = [name for name in identity if name in runs] + metrics
    table = runs[columns].rename(columns=lambda name: name.removeprefix("metrics.").removeprefix("params.").removeprefix("tags.mlflow."))
    st.dataframe(table, hide_index=True, width="stretch", height=680)
    st.download_button("Download complete metrics table", table.to_csv(index=False).encode("utf-8"),
                       "latest_model_metrics.csv", "text/csv")
    st.markdown('<div class="note">Configurations are ranked using cross-validation RMSE. Held-out test scores are not used to select the best model.</div>', unsafe_allow_html=True)


def shap_tab(runs: pd.DataFrame, data: pd.DataFrame) -> None:
    st.subheader("SHAP analysis")
    cv_metric = "metrics.cv_best_rmse"
    if cv_metric not in runs or runs[cv_metric].notna().sum() == 0:
        st.warning("No cross-validation RMSE was logged, so a model cannot be selected for SHAP.")
        return
    best = runs.loc[runs[cv_metric].idxmin()]
    path = model_path(best)
    if not path.exists():
        st.error(f"Fitted model artifact not found: {path.relative_to(ROOT)}")
        return
    ablation = next(item for item in CONFIG["ablation_sets"] if item["name"] == best["params.ablation_name"])
    columns = tuple(get_ablation_features(data, ablation))
    index_file = RESULTS_DIR / "evaluation" / "test_indices.json"
    if index_file.exists():
        indices = tuple(json.loads(index_file.read_text(encoding="utf-8")))
    else:
        from sklearn.model_selection import train_test_split
        _, test = train_test_split(np.arange(len(data)), test_size=CONFIG.get("test_size", 0.3),
                                   random_state=CONFIG.get("seed", 42), shuffle=True)
        indices = tuple(int(index) for index in test)

    st.markdown(f"**Lowest CV RMSE:** {best['params.ablation_name']} | {best['params.model_name']} | {best['params.feature_selection']}")
    st.caption(f"CV RMSE: {best[cv_metric]:.4f} | test R²: {best.get('metrics.test_r2', np.nan):.4f} | test RMSE: {best.get('metrics.test_rmse', np.nan):.4f} | test MAE: {best.get('metrics.test_mae', np.nan):.4f}")
    try:
        with st.spinner("Computing TreeSHAP values for held-out test observations..."):
            transformed, values = shap_data(str(path), columns, indices)
    except Exception as error:
        st.exception(error)
        return

    importance = values.abs().mean().sort_values(ascending=False)
    summary = pd.DataFrame({"Feature": importance.index,
                            "Mean |SHAP value|": importance.values,
                            "Mean signed SHAP value": values.mean().reindex(importance.index).values})
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**Global importance**")
        shown = summary.head(20).sort_values("Mean |SHAP value|")
        fig, ax = plt.subplots(figsize=(8, max(4.5, min(9, len(shown) * 0.3))))
        ax.barh(shown.Feature, shown["Mean |SHAP value|"], color="#147D64")
        ax.set_xlabel("Mean absolute SHAP value")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#DCE4D9", linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with right:
        st.markdown("**SHAP summary (beeswarm)**")
        fig, _ = plt.subplots(figsize=(8, max(4.5, min(9, len(summary) * 0.3))))
        shap.summary_plot(values.to_numpy(), transformed, max_display=20, show=False,
                          plot_size=None, cmap=plt.get_cmap("coolwarm"))
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("### SHAP dependence for the six most important features")
    top_features = importance.head(6).index
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
    for index, feature in enumerate(top_features):
        ax = axes.flat[index]
        feature_values = transformed[feature]
        points = ax.scatter(
            feature_values,
            values[feature],
            c=feature_values,
            cmap="coolwarm",
            s=18,
            alpha=0.75,
        )
        ax.axhline(0, color="#60736D", linestyle="--", linewidth=0.8)
        ax.set_title(feature)
        ax.set_xlabel("Transformed feature value")
        ax.set_ylabel("SHAP value")
        ax.grid(True, linestyle=":", alpha=0.35)
        fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04)
    for index in range(len(top_features), axes.size):
        fig.delaxes(axes.flat[index])
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("### SHAP values for every selected feature")
    st.dataframe(summary.style.format({"Mean |SHAP value|": "{:.5f}", "Mean signed SHAP value": "{:.5f}"}),
                 hide_index=True, width="stretch")
    st.markdown("### Feature ranges at SHAP extremes")
    ranges = []
    for feature in importance.index:
        effect, feature_values = values[feature], transformed[feature]
        n_extreme = max(1, int(len(effect) * 0.05))
        positive, negative = effect.nlargest(n_extreme).index, effect.nsmallest(n_extreme).index
        ranges.append({"Feature": feature,
                       "Feature range at positive SHAP": f"{feature_values.loc[positive].min():.4g} to {feature_values.loc[positive].max():.4g}",
                       "Feature range at negative SHAP": f"{feature_values.loc[negative].min():.4g} to {feature_values.loc[negative].max():.4g}",
                       "Mean positive SHAP": effect.loc[positive].mean(),
                       "Mean negative SHAP": effect.loc[negative].mean()})
    st.dataframe(pd.DataFrame(ranges).style.format({"Mean positive SHAP": "{:.5f}", "Mean negative SHAP": "{:.5f}"}),
                 hide_index=True, width="stretch")


try:
    runs, data = latest_runs(), observations()
except Exception as error:
    st.error("Could not load the configured dataset or MLflow experiment.")
    st.exception(error)
    st.stop()
if runs.empty:
    st.error("No successful MLflow runs match the current configuration. Check the tracking URI, experiment, and configured model/selector names.")
    st.stop()

tabs = st.tabs(
    [
        "Study workflow",
        "Feature selection & collinearity",
        "Model hyperparameters",
        "Evaluation metrics",
        "SHAP analysis",
    ],
    on_change="rerun",
)
with tabs[0]:
    if tabs[0].open:
        workflow_tab(runs, data)
with tabs[1]:
    if tabs[1].open:
        feature_selection_tab(runs, data)
with tabs[2]:
    if tabs[2].open:
        hyperparameters_tab(runs)
with tabs[3]:
    if tabs[3].open:
        metrics_tab(runs)
with tabs[4]:
    if tabs[4].open:
        shap_tab(runs, data)