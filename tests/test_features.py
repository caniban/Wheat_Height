import pandas as pd
import numpy as np

from cropheight.features import compute_vif_and_tolerance, get_selected_features

def test_get_selected_features_all():
    X = pd.DataFrame({
        "PAZ_1": [1, 2, 3],
        "S1_1": [4, 5, 6],
        "TDX_1": [7, 8, 9]
    })
    y = pd.Series([10, 20, 30])

    selected = get_selected_features(X, y, list(X.columns), "ALL", 3)
    assert set(selected) == {"PAZ_1", "S1_1", "TDX_1"}


def test_vif_identifies_correlated_predictors():
    random = np.random.default_rng(42)
    predictor = np.linspace(-1, 1, 100)
    X = pd.DataFrame({
        "correlated_a": predictor,
        "correlated_b": predictor + random.normal(0, 0.01, len(predictor)),
        "independent": random.normal(size=len(predictor)),
    })

    result = compute_vif_and_tolerance(X).set_index("Feature")

    assert result.loc["correlated_a", "VIF"] > 10
    assert result.loc["correlated_b", "VIF"] > 10
    assert result.loc["independent", "VIF"] < 5
    assert result.loc["correlated_a", "Status"] == "High"
    assert np.isclose(
        result.loc["correlated_a", "Tolerance"],
        1 / result.loc["correlated_a", "VIF"],
    )