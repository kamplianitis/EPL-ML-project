import pandas as pd
from sklearn.decomposition import PCA
from typing import Literal


def apply_pca(
    data: list,
    n_components: int = None,
    svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto",
    whiten: bool = False,
    random_state: int = None,
) -> tuple:
    """
    Applies PCA to the data.

    Args:
        data (list): Scaled data.
        n_components (int): Number of principal components.
        svd_solver (str): SVD solver to use ('auto', 'full', 'arpack', or 'randomized').

    Returns:
        tuple: (PCA-transformed data, PCA object)
    """

    pca = PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        whiten=whiten,
        random_state=random_state,
    )
    principal_components = pca.fit_transform(data)
    return principal_components, pca
