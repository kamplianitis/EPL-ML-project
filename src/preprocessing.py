import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from CSV.

    Parameters:
        filepath(str): the path to the desired file that the data will be extracted

    Returns:
        A pandas dataframe that will be used in the processing of the data.
    """

    generated_dataframe = pd.read_csv(filepath)

    return generated_dataframe


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset clearance. The function exists in order to handle every missing value
    existing in the dataset. Also it transforms strings reffering to dates to  datetime
    fields.

    Parameters:
        df (pd.DataFrame): The dataframe that needs clearing

    Retuns:
        A new clear pandas dataframe
    """
    df = df.dropna()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df
