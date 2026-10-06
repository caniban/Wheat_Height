from __future__ import annotations

import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)
    n = 300

    df = pd.DataFrame({
        "PointID": [f"{2021 + (i % 5)}_{i % 10}_{chr(65 + (i % 5))}" for i in range(n)],
        "Crop_Height": rng.normal(110, 20, n)
    })

    for i in range(1, 21):
        df[f"PAZ_{i}"] = rng.normal(10 + i * 0.7, 2, n)
        df[f"S1_{i}"] = rng.normal(15 + i * 0.9, 2, n)
        df[f"TDX_{i}"] = rng.normal(12 + i * 0.8, 2, n)

    df.to_excel("data/crop_heights.xlsx", index=False)
    print("Synthetic dataset generated at data/crop_heights.xlsx")

if __name__ == "__main__":
    main()