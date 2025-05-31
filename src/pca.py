import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Literal


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
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors="ignore")
    numeric_df = df.select_dtypes(include=["number"])

    imputer = SimpleImputer(strategy=impute_strategy)
    imputed_data = imputer.fit_transform(numeric_df)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    return scaled_data, scaler, numeric_df.columns.tolist()


def apply_pca(
    data: list,
    n_components: int = 2,
    svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto",
) -> tuple:
    """
    Applies PCA to the data.

    Args:
        data (array-like): Scaled data.
        n_components (int): Number of principal components.
        svd_solver (str): SVD solver to use ('auto', 'full', 'arpack', or 'randomized').

    Returns:
        tuple: (PCA-transformed data, PCA object)
    """
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    principal_components = pca.fit_transform(data)
    return principal_components, pca


def get_pca_dataframe(pca_data: list) -> pd.DataFrame:
    """
    Creates a DataFrame from PCA-transformed data.

    Args:
        pca_data (array-like): PCA output.

    Returns:
        pd.DataFrame: PCA result as DataFrame.
    """
    columns = [f"PC{i+1}" for i in range(pca_data.shape[1])]
    return pd.DataFrame(pca_data, columns=columns)
