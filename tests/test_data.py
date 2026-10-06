import pandas as pd

from cropheight.data import get_ablation_features, get_feature_columns

def test_get_feature_columns():
    df = pd.DataFrame({
        "PointID": ["2021_1_A", "2021_2_B"],
        "Crop_Height": [80, 90],
        "PAZ_1": [1.0, 2.0],
        "S1_1": [3.0, 4.0],
        "TDX_1": [5.0, 6.0],
    })

    cols = get_feature_columns(df)
    assert set(cols) == {"PAZ_1", "S1_1", "TDX_1"}

def test_get_ablation_features():
    df = pd.DataFrame({
        "PointID": ["2021_1_A", "2021_2_B"],
        "Crop_Height": [80, 90],
        "PAZ_1": [1.0, 2.0],
        "S1_1": [3.0, 4.0],
        "TDX_1": [5.0, 6.0],
    })

    ablation_cfg = {"name": "Only X-Band (PAZ & TDX)", "columns_prefix": ["PAZ_", "TDX_"]}
    selected = get_ablation_features(df, ablation_cfg)
    assert set(selected) == {"PAZ_1", "TDX_1"}