from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from typing import Literal, Tuple


def preprocess_data(
    df: pd.DataFrame, target_column: str, scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, StandardScaler | None]:
    """
    Splits the dataframe into features and target, applies scaling if specified.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Column to use as target variable.
        scale (bool): Whether to apply StandardScaler.

    Returns:
        Tuple: (X, y, scaler) where scaler is None if scaling is False.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X.values

    return X, y, scaler


def train_naive_bayes(
    X: np.ndarray,
    y: pd.Series,
    model_type: Literal["gaussian", "multinomial", "bernoulli"] = "gaussian",
    alpha: float = 1.0,
):
    """
    Trains a Naive Bayes model.

    Args:
        X (np.ndarray): Features.
        y (pd.Series): Target.
        model_type (str): One of "gaussian", "multinomial", "bernoulli".
        alpha (float): Laplace smoothing (used for multinomial and bernoulli).

    Returns:
        model: Trained classifier.
    """
    if model_type == "gaussian":
        model = GaussianNB()
    elif model_type == "multinomial":
        model = MultinomialNB(alpha=alpha)
    elif model_type == "bernoulli":
        model = BernoulliNB(alpha=alpha)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model and prints metrics.
    """
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, average="macro"))
    print("Recall:", recall_score(y_test, predictions, average="macro"))
    print("F1 Score:", f1_score(y_test, predictions, average="macro"))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
