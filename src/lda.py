import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Literal


def apply_lda(
    X: list,
    Y: list,
    n: int | None,
    solver: Literal["svd", "lsqr", "eigen"],
    store_cov: bool,
    shirnkage: float | str,
) -> tuple:
    """
    Applies Linear Discriminant Analysis (LDA) to the given data.

    LDA is a supervised dimensionality reduction technique that projects data into a
    lower-dimensional space while preserving class separability.

    Parameters:
        X (list): Feature matrix (typically preprocessed and scaled).
        Y (list): Target labels corresponding to each sample in X.
        n (int | None): Number of components to retain. If None, automatically inferred.
        solver (Literal["svd", "lsqr", "eigen"]): Solver used to compute LDA.
            - 'svd': Fastest; does not support shrinkage.
            - 'lsqr': Uses least squares; supports shrinkage.
            - 'eigen': Uses eigenvalue decomposition; supports shrinkage.
        store_cov (bool): Whether to store the covariance matrix of each class.
        shirnkage (float | str): Shrinkage parameter. Can be a float (0.0–1.0),
            'auto' (for automatic shrinkage), or None. Only used with 'lsqr' and 'eigen' solvers.

    Returns:
        tuple: A tuple containing:
            - X_lda (np.ndarray): LDA-transformed feature matrix.
            - lda (LinearDiscriminantAnalysis): The fitted LDA model instance.
    """

    lda = LinearDiscriminantAnalysis(
        n_components=n,
        solver=solver,
        store_covariance=store_cov,
        shrinkage=shirnkage,
    )
    X_lda = lda.fit_transform(X, Y)

    return X_lda, lda
