from pathlib import Path

from cropheight.config import ExperimentConfig

def test_config_loads_yaml():
    cfg = ExperimentConfig.from_yaml(Path("configs/base.yaml"))
    assert cfg.seed == 42
    assert cfg.cv_folds == 5
    assert "RandomForest" in cfg.models
    assert "ALL" in cfg.feature_selection_methods