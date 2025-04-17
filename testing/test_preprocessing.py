import pytest
from unittest.mock import patch
import pandas as pd

from src.preprocessing import load_data


class TestLoadData:
    @patch("pandas.read_csv")
    def test_load_data_success(self, mock_read_csv):
        mock_df = pd.DataFrame({"column1": [1, 2], "column2": [3, 4]})
        mock_read_csv.return_value = mock_df
        result = load_data("path/to/file.csv")
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("pandas.read_csv")
    def test_load_data_special_characters(self, mock_read_csv):
        mock_df = pd.DataFrame({"col@umn1": [1, 2], "col#umn2": [3, 4]})
        mock_read_csv.return_value = mock_df
        result = load_data("path/to/special_characters.csv")
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("pandas.read_csv")
    def test_load_data_numeric(self, mock_read_csv):
        mock_df = pd.DataFrame({"numbers": [1, 2, 3, 4]})
        mock_read_csv.return_value = mock_df
        result = load_data("path/to/numeric.csv")
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("pandas.read_csv")
    def test_load_data_non_existent_file(self, mock_read_csv):
        mock_read_csv.side_effect = FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_data("path/to/non_existent.csv")

    @patch("pandas.read_csv")
    def test_load_data_empty_file(self, mock_read_csv):
        mock_df = pd.DataFrame()
        mock_read_csv.return_value = mock_df
        result = load_data("path/to/empty.csv")
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("pandas.read_csv")
    def test_load_data_missing_values(self, mock_read_csv):
        mock_df = pd.DataFrame({"column1": [1, None], "column2": [3, 4]})
        mock_read_csv.return_value = mock_df
        result = load_data("path/to/missing_values.csv")
        pd.testing.assert_frame_equal(result, mock_df)
