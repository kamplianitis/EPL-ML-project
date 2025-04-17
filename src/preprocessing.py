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
