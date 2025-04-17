import pytest
from unittest.mock import patch
import pandas as pd

from src.preprocessing import load_data
from src.preprocessing import clean_data


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


class TestCleanData:

    def test_drops_rows_with_nan_values(self):
        import numpy as np

        df = pd.DataFrame({"col1": [1, 2, np.nan, 4], "col2": [5, np.nan, 7, 8]})
        result = clean_data(df)
        assert len(result) == 2
        assert result.iloc[0, 0] == 1
        assert result.iloc[0, 1] == 5
        assert not result.isna().any().any()

    def test_empty_dataframe_handling(self):
        from src.preprocessing import clean_data

        empty_df = pd.DataFrame()

        result = clean_data(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert result.empty

    def test_date_column_conversion(self):
        data = {"date": ["2021-01-01", "2021-02-01"], "value": [10, 20]}
        df = pd.DataFrame(data)
        result = clean_data(df)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_returns_new_dataframe(self):
        data = {"date": ["2021-01-01", "2021-02-01"], "value": [10, 20]}
        df = pd.DataFrame(data)
        original_df = df.copy()
        result = clean_data(df)

        assert not result.equals(original_df)
        assert df.equals(original_df)

    def test_no_missing_values(self):
        import pandas.testing as pdt

        data = {"date": ["2021-01-01", "2021-02-01"], "value": [10, 20]}
        df = pd.DataFrame(data)

        expected = df.copy()
        expected["date"] = pd.to_datetime(expected["date"])
        result = clean_data(df)

        pdt.assert_frame_equal(result, expected)
