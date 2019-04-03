import os
import pandas as pd
import numpy as np
from typing import Tuple

from torch.utils.data import Dataset
from torch.tensor import Tensor


class CustomDataset(Dataset):
    """Custom fraudalent traffic detection dataset"""

    TARGET_COLUMN = "is_attributed"

    def __init__(self, file_name: str = "train.csv",
                 root_dir: str = os.path.join(os.getcwd(), "data")) -> None:
        """
        Args:
            file_name (string): Name of the file to be loaded.
            root_dir (string): Directory containing all the necessary datasets.

        Returns:
            None
        """

        assert set([file_name]) & set(os.listdir(root_dir)), \
            f"file_name passed as an argument {file_name} " \
            f"is not in given {root_dir} directory, {os.listdir(root_dir)} are."

        data = pd.read_csv(os.path.join(root_dir, file_name), header=0, sep=",", nrows=500)
        data = self._preprocess_data(data)
        self.len = len(data)
        self.x_data = data.drop([self.TARGET_COLUMN], axis=1).values
        self.y_data = data["is_attributed"].values

        assert self.len == len(self.x_data) == len(self.y_data), \
            f"length mismatch, whole dataset's length is {self.len}, " \
            f"whereas x_data's length is {len(self.x_data)} and y_data's - {len(self.y_data)}."

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            data (pd.DataFrame): data read from *.csv file
        Returns:
            data (pd.DataFrame): preprocessed data
        """

        data = self._handle_time_categorical_data(data)
        all_columns = data.columns

        # Drop columns which nans make more than 65% of all datapoints, since they're irrelevant
        columns_to_drop = [attribute for attribute in all_columns if
                           len([x for x in data[attribute].isna() if x is not False]) / len(
                               data) * 100.0 >= 65.0]
        print(f"Columns to drop, because nans share has exceeded threshold: {columns_to_drop}")

        target_corr_matrix = data.corr()[self.TARGET_COLUMN]
        columns_to_drop.extend([attribute for attribute in target_corr_matrix.index
                                if data.corr()[self.TARGET_COLUMN][attribute] < 0.01])
        print(f"Previous columns to drop, extended by columns which correlation "
              f"to the target column is lower then threshold: {columns_to_drop}")
        data = data.drop(columns_to_drop, axis=1)
        return data

    @staticmethod
    def _handle_time_categorical_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            data (pd.DataFrame): data with datetime categorical attribute
        Returns:
            data (pd.DataFrame): data with one-hot encoded datetime information
        """
        data["click_time_date"] = data["click_time"].apply(lambda l: l[:l.find(' ')])
        data["click_time_year"] = data["click_time_date"].apply(lambda l: int(l[:l.find('-')]))
        data["click_time_month"] = (data["click_time_date"]
                                    .apply(lambda l: int(l[l.find('-') + 1:l.rfind('-')])))
        data["click_time_day"] = data["click_time_date"].apply(lambda l: int(l[l.rfind('-') + 1:]))

        data["click_time"] = data["click_time"].apply(lambda l: l[l.find(' ') + 1:])
        data["click_time_hours"] = (data["click_time"].apply(lambda l: int(l[:l.find(':')])))
        data["click_time_mins"] = (data["click_time"]
                                   .apply(lambda l: int(l[l.find(':') + 1:l.rfind(':')])))
        data["click_time_secs"] = data["click_time"].apply(lambda l: int(l[l.rfind(':') + 1:]))
        # ranged between 0 and 47 - corresponds to half-hours
        data["click_time"] = (data["click_time_hours"] * 60.0 +
                              data["click_time_mins"] +
                              data["click_time_secs"] / 60.0) / 30.0
        data["click_time"] = data["click_time"].apply(lambda l: round(l, 2))

        click_time_dummies = pd.get_dummies(data["click_time"],
                                            prefix="click_time_hour_halves")
        click_time_year_dummies = pd.get_dummies(data["click_time_year"],
                                                 prefix="click_time_year")
        click_time_month_dummies = pd.get_dummies(data["click_time_month"],
                                                  prefix="click_time_month")
        click_time_day_dummies = pd.get_dummies(data["click_time_day"],
                                                prefix="click_time_day")

        data.drop(["click_time_date", "click_time_year",
                   "click_time_month", "click_time_day",
                   "click_time", "click_time_hours",
                   "click_time_mins", "click_time_secs"],
                  axis=1, inplace=True)
        data_w_ohe_time_categorical_attributes = pd.concat(
            [data, click_time_dummies, click_time_year_dummies,
             click_time_month_dummies, click_time_day_dummies], axis=1)
        return data_w_ohe_time_categorical_attributes

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return self.len

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """
        Args:
            idx: Specifies position of dataset to be returned.

        Returns:
            Tuple[Tensor, pd.DataFrame]: samples of x_data and y_data
        """
        return self.x_data[idx], self.y_data[idx]
