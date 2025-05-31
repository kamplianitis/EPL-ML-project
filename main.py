import os
import pandas as pd
from src.pca import preprocess_data
from src.pca import apply_pca
from src.pca import get_pca_dataframe


def csv_to_dataset(file_name: str) -> pd.DataFrame:
    """
    Check the file's extension to ensure that the file given is a .csv file. Locates the
    file in the ./data directory and then passes the file to pandas in order to return
    a DataFrame object.
    """

    if not file_name.lower().endswith(".csv"):
        raise ValueError(f"Invalid file type: '{file_name}' is not a CSV file")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    full_path = os.path.abspath(os.path.join(data_dir, file_name))

    if os.path.isfile(full_path):
        return full_path
    else:
        raise FileNotFoundError(f"'{file_name}' not found in data directory")


def main():
    try:
        file_path = csv_to_dataset("PremierLeague.csv")
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return -1

    df = pd.read_csv(filepath_or_buffer=file_path)

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

    data, scaler, feature_names = preprocess_data(df, exclude_columns=exclude_columns)

    n_components = 6
    svd_solver = "randomized"

    principal_components, pca = apply_pca(
        data, n_components=n_components, svd_solver=svd_solver
    )

    pca_df = get_pca_dataframe(principal_components)

    print("\nExplained Variance Ratio:")
    print(pca.explained_variance_ratio_)

    print("\nPCA Result Preview:")
    print(pca_df.head())


if __name__ == "__main__":
    main()
