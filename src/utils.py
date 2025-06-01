import pandas as pd
from pathlib import Path


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


# TODO: move this to utils
def preprocess_data(
    df: pd.DataFrame,
    exclude_columns: list = None,
    impute_strategy="mean",
) -> tuple:
    """
    Removes non-numeric columns and standardizes the data.

    Args:
        df (pd.DataFrame): Input data.
        exclude_columns (list): Columns to exclude before PCA.

    Returns:
        tuple: (preprocessed data array, StandardScaler object, feature names)
    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors="ignore")
    numeric_df = df.select_dtypes(include=["number"])

    imputer = SimpleImputer(strategy=impute_strategy)
    imputed_data = imputer.fit_transform(numeric_df)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    return scaled_data, scaler, numeric_df.columns.tolist()
