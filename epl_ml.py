from typing import Optional
import typer


app = typer.Typer(add_completion=False)


@app.command()
def pca(
    dataset: str = typer.Option(..., exists=True, help="The name of the dataset"),
    n: Optional[int] = typer.Option(
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
        metavar="{'auto','full','arpack','randomized'}",
        prompt=False,
        help="Singular Value Decomposition method.",
    ),
    randon_state: Optional[int] = typer.Option(
        None, "--random-state", help="random number generation for reproducibility."
    ),
    all: Optional[bool] = typer.Option(False, "--all", help="Run all the simulations."),
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

    if not all:
        exclude_columns = [
            "MarketMaxHomeTeam",
            "MarketMaxDraw",
            "MarketMaxAwayTeam",
            "MarketAvgHomeTeam",
            "MarketAvgDraw",
            "MarketAvgAwayTeam",
            "MarketMaxOver2.5Goals",
            "MarketMaxUnder2.5Goals",
            "MarketAvgOver2.5Goals",
            "MarketAvgUnder2.5Goals",
        ]

        data, _, _ = preprocess_data(df=df, exclude_columns=exclude_columns)

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

        print("\nPCA Result Preview:")
        print(pca_df.head())


@app.command()
def lda(
    dataset: str = typer.Option(..., exists=True, help="The name of the dataset"),
):
    pass


if __name__ == "__main__":
    app()
