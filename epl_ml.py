from typing import Optional
import typer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

app = typer.Typer(add_completion=False)


@app.command()
def pca(
    dataset: str = typer.Option(..., exists=True, help="The name of the dataset"),
    n: Optional[float] = typer.Option(
        None, "--n", help="The number of principal components to retain"
    ),
    whiten: Optional[bool] = typer.Option(
        False,
        "--whiten/--no-whiten",
        help="Data won't be applied to standardize variance.",
    ),
    svd: Optional[str] = typer.Option(
        "auto",
        "--svd",
        case_sensitive=False,
        show_choices=True,
        metavar="['auto','full','arpack','randomized']",
        prompt=False,
        help="Singular Value Decomposition method.",
    ),
    randon_state: Optional[int] = typer.Option(
        None, "--random-state", help="random number generation for reproducibility."
    ),
):

    import pandas as pd
    from src.utils import csv_to_dataset
    from src.utils import preprocess_data
    from src.pca import apply_pca
    from src.utils import get_dataframe

    try:
        file_path = csv_to_dataset(dataset)
    except (FileNotFoundError, ValueError) as e:
        raise e

    df = pd.read_csv(filepath_or_buffer=file_path)

    data, _, _ = preprocess_data(df=df, exclude_columns=[])

    principal_components, pca = apply_pca(
        data,
        n_components=n,
        svd_solver=svd,
        whiten=whiten,
        random_state=randon_state,
    )

    pca_df = get_dataframe(data=principal_components, col_name="PC")

    print("\nExplained Variance Ratio:")
    print(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 4))
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        marker="o",
    )
    plt.title("Explained Variance Ratio by Principal Components")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca_explained_variance.png")
    plt.close()

    print("\nPCA Result Preview:")
    print(pca_df)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, len(explained_variance) + 1),
        cumulative_variance,
        marker="o",
        linestyle="--",
    )
    plt.title("Cumulative Explained Variance by Principal Components")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca_scatter_1.png")
    plt.close()


@app.command()
def lda(
    dataset: str = typer.Option(..., exists=True, help="The name of the dataset"),
    n: Optional[int] = typer.Option(
        None, "--n", help="The number of linear discriminants to retain."
    ),
    solver: Optional[str] = typer.Option(
        "svd",
        "--solver",
        case_sensitive=False,
        show_choices=True,
        metavar="['svd','lsqr', 'eigen']",
        prompt=False,
        help="The solving method option.",
    ),
    store_cov: Optional[bool] = typer.Option(
        False,
        "--store-cov/ --no-store-cov",
        help="Whether the covariance matrix will be stored.",
    ),
    shrinkage: Optional[float] = typer.Option(
        None,
        "--shrink",
        help="A shrinkage method to improve the estimation of the covariance matrix.",
    ),
):
    import pandas as pd
    from src.utils import csv_to_dataset
    from src.utils import preprocess_data
    from src.lda import apply_lda
    from src.utils import get_dataframe

    try:
        file_path = csv_to_dataset(dataset)
    except (FileNotFoundError, ValueError) as e:
        raise e

    df = pd.read_csv(filepath_or_buffer=file_path)

    label_collumn = "FullTimeHomeTeamGoals"

    x, y, _, _ = preprocess_data(
        df=df,
        exclude_columns=[],
        label_column=label_collumn,
        supervised=True,
    )

    X_lda, lda_model = apply_lda(
        X=x,
        Y=y,
        n=n,
        solver=solver,
        store_cov=store_cov,
        shirnkage=shrinkage,
    )

    lda_df = get_dataframe(data=X_lda, col_name="LD")

    print("\nExplained Variance Ratio:")
    print(lda_model.explained_variance_ratio_)
    cumulative_var = np.cumsum(lda_model.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(lda_model.explained_variance_ratio_) + 1),
        cumulative_var,
        marker="o",
        linestyle="--",
    )
    plt.title("Cumulative Explained Variance (LDA)")
    plt.xlabel("Number of Linear Discriminants")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lda_scatter_1.png")
    plt.close()

    print("\nLDA Result Preview:")
    print(lda_df)


@app.command()
def naiveBayes(
    dataset: str = typer.Option(..., exists=True, help="The name of the dataset"),
    model_type: Optional[str] = typer.Option(
        "gaussian",
        "--model",
        case_sensitive=False,
        show_choices=True,
        metavar="['gaussian','multinomial', 'bernoulli']",
        prompt=False,
        help="The solving method option.",
    ),
    alpha: Optional[float] = typer.Option(1, "--alpha", help="Smoothing."),
):
    import pandas as pd
    from src.NaiveBayes import preprocess_data, train_naive_bayes, evaluate_model
    from src.utils import csv_to_dataset
    from sklearn.model_selection import train_test_split

    try:
        file_path = csv_to_dataset(dataset)
    except (FileNotFoundError, ValueError) as e:
        raise e

    df = pd.read_csv(file_path)

    exclude_columns = [
        "MatchID",
        "Season",
        "Date",
        "Time",
        "HomeTeam",
        "AwayTeam",
        "FullTimeResult",
        "HalfTimeResult",
        "Referee",
    ]

    df = df.drop(columns=exclude_columns, errors="ignore")
    df = df.dropna(axis=1)

    target_column = "HomeTeamPoints"

    X, y, _ = preprocess_data(df, target_column=target_column, scale=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    model = train_naive_bayes(X_train, y_train, model_type="gaussian")

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    app()
