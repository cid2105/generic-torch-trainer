import pytest


@pytest.fixture
def small_config():
    return {
        "data": {
            "use_toy_data": True,
            "toy_data": {"n": 200, "seed": 0},
            "input_path": None,
            "target_col": "label",
        },
        "split": {"test_size": 0.2, "random_state": 0, "stratify": True},
        "preprocessing": {
            "numeric": {
                "imputer": {"strategy": "median"},
                "scaler": {"enabled": True},
            },
            "categorical": {
                "imputer": {"strategy": "constant", "fill_value": "missing"},
                "onehot": {"handle_unknown": "ignore", "sparse_output": False},
            },
        },
        "model": {"name": "BinaryClassifier", "hidden_dims": [16], "dropout": 0.0},
        "training": {
            "batch_size": 64,
            "epochs": 1,
            "shuffle_train": True,
            "device": "cpu",
            "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0},
            "loss": {"name": "BCEWithLogitsLoss"},
        },
        "evaluation": {"threshold": 0.5},
        "inference": {"drop_target_col_if_present": True},
        "output": {"dir": "outputs"},
    }
