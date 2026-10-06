# Wheat Height Model Laboratory

Streamlit dashboard for the SAR crop-height regression experiments. It reads the configured dataset, the MLflow SQLite tracking database, and the fitted model pipelines under `results/artifacts/`.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository's branch and select `streamlit_app.py` as the main file.
3. Keep the required runtime files in the repository: `data/crop_heights.xlsx`, `mlruns.db`, `results/artifacts/`, and `results/evaluation/test_indices.json`.
4. Set the Python version to 3.14 in the app's advanced settings to match the environment used to fit the saved pipelines.

`mlruns/` contains bulky local tracking artifacts and is intentionally excluded from Git; the dashboard reads run parameters and metrics from `mlruns.db` and computes SHAP explanations from the saved model.

The repository includes the source dataset and fitted models. Use a **private** GitHub repository unless you have permission to publish these files.