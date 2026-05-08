import pandas as pd
from sklearn.model_selection import train_test_split

from data import load_dataframe
from preprocess import build_preprocessor, preprocess_data


def test_preprocess_returns_correct_shapes(small_config):
    df = load_dataframe(small_config)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=0)

    X_train, y_train, X_val, y_val, _, metadata = preprocess_data(
        train_df,
        val_df,
        "label",
        small_config,
    )

    assert X_train.shape[0] == len(train_df)
    assert X_val.shape[0] == len(val_df)
    assert X_train.shape[1] == metadata["input_dim"]
    assert X_val.shape[1] == metadata["input_dim"]
    assert y_train.shape == (len(train_df),)
    assert y_val.shape == (len(val_df),)
    assert "country" in metadata["categorical_cols"]
    assert "age" in metadata["numeric_cols"]


def test_categorical_detection_handles_object_and_string_dtypes(small_config):
    df = pd.DataFrame(
        {
            "num_int": [1, 2, 3],
            "num_float": [1.0, 2.0, 3.0],
            "str_obj": pd.array(["a", "b", "c"], dtype="object"),
            "str_pandas": pd.array(["x", "y", "z"], dtype="string"),
            "label": [0, 1, 0],
        }
    )

    _, _, numeric_cols, categorical_cols = build_preprocessor(df, "label", small_config)

    assert set(numeric_cols) == {"num_int", "num_float"}
    assert set(categorical_cols) == {"str_obj", "str_pandas"}
