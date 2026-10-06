from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.replace("TSX_", "TDX_")
    if "PointID" in df.columns:
        try:
            df[["Year", "Visit", "Point"]] = df["PointID"].str.split("_", expand=True)
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        except Exception:
            pass

    return df

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(("PAZ_", "S1", "TDX_"))]

def get_target_series(df: pd.DataFrame) -> pd.Series:
    return df["Crop_Height"]

def get_ablation_features(df: pd.DataFrame, ablation_config: dict) -> List[str]:
    feature_cols = get_feature_columns(df)
    prefixes = ablation_config.get("columns_prefix", [])
    if "Combined (C-Band + X-Band)" in ablation_config.get("name", ""):
        return feature_cols

    selected = []
    for col in feature_cols:
        if any(col.startswith(prefix) for prefix in prefixes):
            selected.append(col)
    return selected

def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["Crop_Height"]
    return X, y