.PHONY: install test lint train explain clean

install:
    python -m pip install -U pip
    pip install -e .
    pip install pytest ruff pre-commit

test:
    pytest -q

lint:
    ruff check .

train:
    python -m src.cropheight.train --config configs/base.yaml --retrain

explain:
    python -m src.cropheight.explain --model-path results/artifacts/.../model.joblib --data-path data/crop_heights.xlsx

clean:
    rm -rf results mlruns .pytest_cache