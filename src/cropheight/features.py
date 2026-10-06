from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import shap

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


class SHAPRecursiveFeatureEliminator(BaseEstimator, TransformerMixin):
    """Recursively remove the least important feature using RF mean |SHAP|."""

    def __init__(
        self,
        estimator=None,
        n_features_to_select=19,
        random_state=42,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state

    @staticmethod
    def _mean_absolute_shap(shap_values, n_features):
        if isinstance(shap_values, list):
            values = np.stack([np.asarray(value) for value in shap_values], axis=0)
            importance = np.abs(values).mean(axis=tuple(range(values.ndim - 1)))
        else:
            values = np.asarray(
                shap_values.values if hasattr(shap_values, "values") else shap_values
            )
            if values.ndim == 2:  # samples × features
                importance = np.abs(values).mean(axis=0)
            elif values.ndim == 3:  # samples × features × outputs
                importance = np.abs(values).mean(axis=(0, 2))
            else:
                raise ValueError(f"Unexpected SHAP array shape: {values.shape}")

        if len(importance) != n_features:
            raise ValueError(
                f"SHAP returned {len(importance)} importances for {n_features} features."
            )
        return importance

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y)

        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional numeric array.")
        if X_array.shape[1] == 0:
            raise ValueError("SHAP-RFE cannot fit with zero features.")

        target_count = int(self.n_features_to_select)
        if not 1 <= target_count <= X_array.shape[1]:
            raise ValueError(
                "n_features_to_select must be between 1 and the input feature count."
            )

        active = np.arange(X_array.shape[1])
        base_estimator = (
            self.estimator
            if self.estimator is not None
            else RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
            )
        )

        while len(active) > target_count:
            fitted_estimator = clone(base_estimator)
            fitted_estimator.fit(X_array[:, active], y_array)

            explainer = shap.TreeExplainer(fitted_estimator)
            values = explainer.shap_values(
                X_array[:, active],
                check_additivity=False,
            )
            importance = self._mean_absolute_shap(values, len(active))

            # Remove the least important feature, then recompute SHAP.
            active = np.delete(active, int(np.argmin(importance)))

        self.support_ = np.zeros(X_array.shape[1], dtype=bool)
        self.support_[active] = True
        self.n_features_in_ = X_array.shape[1]
        self.selected_features_ = active
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        X_array = np.asarray(X)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count differs from the fitted feature count.")
        return X_array[:, self.support_]

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()


def build_feature_selection_pipeline(
    model,
    fs_method: str,
    n_features: int,
    random_state: int = 42,
) -> Pipeline:
    steps = [("scaler", StandardScaler())]

    if fs_method == "ALL":
        pass
    elif fs_method == "MI":
        steps.append((
            "fs",
            SelectKBest(score_func=mutual_info_regression, k=n_features),
        ))
    elif fs_method == "LASSO":
        steps.append((
            "fs",
            SelectFromModel(
                LassoCV(cv=5, random_state=random_state),
                max_features=n_features,
            ),
        ))
    elif fs_method == "SHAP_RFE":
        rf_selector = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )
        steps.append((
            "fs",
            SHAPRecursiveFeatureEliminator(
                estimator=rf_selector,
                n_features_to_select=n_features,
                random_state=random_state,
            ),
        ))
    else:
        raise ValueError(f"Unsupported feature-selection method: {fs_method}")

    steps.append(("model", model))
    return Pipeline(steps)

def get_selected_features(X: pd.DataFrame, y: pd.Series, feature_cols: List[str], fs_method: str, n_features: int) -> List[str]:
    if fs_method == "ALL":
        return list(feature_cols)

    dummy_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    pipe = build_feature_selection_pipeline(dummy_model, fs_method, n_features)
    pipe.fit(X, y)

    fs_step = pipe.named_steps.get("fs")
    if fs_step is None:
        return list(feature_cols)

    mask = fs_step.get_support()
    return np.array(feature_cols)[mask].tolist()

def compute_vif_and_tolerance(X: pd.DataFrame) -> pd.DataFrame:
    numeric = X.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median())
    varying_columns = [column for column in numeric if numeric[column].nunique() > 1]
    design = (
        add_constant(numeric[varying_columns], has_constant="add").to_numpy(dtype=float)
        if varying_columns
        else np.empty((len(numeric), 0))
    )
    vif_rows = []
    for feature in numeric.columns:
        if feature not in varying_columns:
            vif = np.nan
            tol = np.nan
        else:
            index = varying_columns.index(feature) + 1
            vif = float(variance_inflation_factor(design, index))
            tol = 1.0 / vif if np.isfinite(vif) and vif > 0 else 0.0 if np.isinf(vif) else np.nan

        vif_rows.append({
            "Feature": feature,
            "VIF": vif,
            "Tolerance": tol
        })

    vif_df = pd.DataFrame(vif_rows)
    vif_df["Status"] = np.where(vif_df["VIF"] > 10, "High", np.where(vif_df["VIF"] > 5, "Moderate", "Low"))
    return vif_df

def compute_correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    return X.corr().abs()

def export_multicollinearity_report(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    output_dir: str | Path,
    stem: str = "multicollinearity_report"
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corr_before = compute_correlation_matrix(X_before)
    corr_after = compute_correlation_matrix(X_after)

    vif_before = compute_vif_and_tolerance(X_before)
    vif_after = compute_vif_and_tolerance(X_after)

    with pd.ExcelWriter(output_dir / f"{stem}.xlsx") as writer:
        corr_before.to_excel(writer, sheet_name="Correlation_Before_FS")
        corr_after.to_excel(writer, sheet_name="Correlation_After_FS")
        vif_before.to_excel(writer, sheet_name="VIF_TOL_Before_FS", index=False)
        vif_after.to_excel(writer, sheet_name="VIF_TOL_After_FS", index=False)

    return None