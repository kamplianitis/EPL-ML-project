import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Literal


def csv_to_dataset(file_name: str) -> pd.DataFrame:
    """
    Check the file's extension to ensure that the file given is a .csv file. Locates the
    file in the ./data directory and then passes the file to pandas in order to return
    a DataFrame object.
    """

    import os

    if not file_name.lower().endswith(".csv"):
        raise ValueError(f"Invalid file type: '{file_name}' is not a CSV file")

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    full_path = data_dir / file_name

    if os.path.isfile(full_path):
        return full_path
    else:
        raise FileNotFoundError(f"'{file_name}' not found in data directory")


def preprocess_data(
    df: pd.DataFrame,
    label_column: str | None = None,
    exclude_columns: list = None,
    impute_strategy: str = "mean",
    supervised: bool = False,
) -> (
    tuple[np.ndarray, StandardScaler, list[str]]
    | tuple[np.ndarray, np.ndarray, StandardScaler, list[str]]
):
    """
    Removes non-numeric columns and standardizes the data.

    Parameters:
        df (pd.DataFrame): Input data.
        exclude_columns (list): Columns to exclude before PCA.
        input_strategy (str): The strategy to be followed (Default: mean)
        supervised (bool): Determines whether the processing will be for supervised or
            unsupervised data.

    Returns:
        A tuple based on the needs of data processing
    """

    from sklearn.impute import SimpleImputer

    if not supervised:
        if exclude_columns:
            df = df.drop(columns=exclude_columns, errors="ignore")
        numeric_df = df.select_dtypes(include=["number"])
    else:
        exclude_columns = set(exclude_columns + [label_column])
        features_df = df.drop(columns=exclude_columns, errors="ignore")
        numeric_df = features_df.select_dtypes(include=["number"])

    imputer = SimpleImputer(strategy=impute_strategy)
    imputed_data = imputer.fit_transform(numeric_df)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    if not supervised:
        return scaled_data, scaler, numeric_df.columns.tolist()

    y = df[label_column].values
    return scaled_data, y, scaler, numeric_df.columns.tolist()


def get_dataframe(data: list, col_name: Literal["PC", "LD"]) -> pd.DataFrame:
    """
    Creates a DataFrame from PCA-transformed data.

    Parameters:
        data (list): PCA/LDA output.
        col_name (Literal["PC", "LD"]): PC or LD depending on the proccess.

    Returns:
        pd.DataFrame: PCA/LDA result as DataFrame.
    """

    columns = [f"{col_name}{i+1}" for i in range(data.shape[1])]
    return pd.DataFrame(data, columns=columns)
