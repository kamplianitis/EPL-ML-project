import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def train_lse_model(X_train, y_train) -> LinearRegression:
    """
    Trains a Least Squares Error (Linear Regression) model.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training targets.

    Returns:
        LinearRegression: Trained linear model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_lse_model(model, X_test, y_test):
    """
    Evaluates the trained LSE model using MSE and R^2.

    Args:
        model (LinearRegression): Trained model.
        X_test (array-like): Test features.
        y_test (array-like): Test targets.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
