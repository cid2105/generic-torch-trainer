from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    df: pd.DataFrame,
    target_col: str,
    config: dict[str, Any],
):
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    pp = config["preprocessing"]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy=pp["numeric"]["imputer"]["strategy"])),
    ]
    if pp["numeric"]["scaler"]["enabled"]:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy=pp["categorical"]["imputer"]["strategy"],
                fill_value=pp["categorical"]["imputer"]["fill_value"],
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown=pp["categorical"]["onehot"]["handle_unknown"],
                sparse_output=pp["categorical"]["onehot"]["sparse_output"],
            ),
        ),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", Pipeline(categorical_steps), categorical_cols),
        ]
    )

    return preprocessor, feature_cols, numeric_cols, categorical_cols


def preprocess_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    config: dict[str, Any],
):
    preprocessor, feature_cols, numeric_cols, categorical_cols = build_preprocessor(
        train_df,
        target_col,
        config,
    )

    X_train = preprocessor.fit_transform(train_df[feature_cols]).astype(np.float32)
    X_val = preprocessor.transform(val_df[feature_cols]).astype(np.float32)

    y_train = train_df[target_col].values.astype(np.float32)
    y_val = val_df[target_col].values.astype(np.float32)

    metadata = {
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "input_dim": X_train.shape[1],
    }

    return X_train, y_train, X_val, y_val, preprocessor, metadata
