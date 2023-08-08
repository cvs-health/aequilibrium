# Copyright 2022 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DataSet:

    """Defines a valid aequilibrium dataset

    Args:
        predictors (pd.DataFrame): Predictors for model(s)
        target (pd.DataFrame): Response variable for model(s)
        target_distribution (pd.Series): The distribution of the response variable
        num_predictors (int): The number of predictors being used
        num_samples (int): The number of samples available
        balanced (bool): True if the dataset is balanced, False otherwise

    Returns:
        None
    """

    # Required properties
    predictors: pd.DataFrame
    target: pd.Series

    # Automatically populated properties
    target_distribution: pd.Series = field(init=False)
    num_predictors: int = field(init=False)
    num_samples: int = field(init=False)
    balanced: bool = field(init=False)

    def __post_init__(self):
        # Predictors must be a DataFrame
        if not isinstance(self.predictors, pd.DataFrame):
            raise TypeError(
                f"aequilibrium requires the DataSet parameter `predictors` to be a populated Pandas DataFrame. Instead, you passed an object of type `{type(self.predictors)}`"
            )

        # Predictors must be populated
        if self.predictors.empty:
            raise ValueError(
                f"aequilibrium requires the DataSet parameter `predictors` to be a populated Pandas DataFrame. Instead, you passed a DataFrame with shape `{self.predictors.shape}`"
            )

        # Response must be a Series
        if not isinstance(self.target, pd.Series):
            raise TypeError(
                f"aequilibrium requires the DataSet parameter `target` to be a populated Pandas Series. Instead, you passed an object of type `{type(self.target)}`"
            )

        # Response must be populated
        if self.target.empty:
            raise ValueError(
                f"aequilibrium requires the DataSet parameter `target` to be a populated Pandas Series. Instead, you passed a Series with shape `{self.target.size}`"
            )

        # Guarantee that provided data has the same row count
        if self.predictors.shape[0] != self.target.size:
            raise ValueError(
                f"aequilibrium requires the DataSet parameters `predictors` and `target` to have the same number of rows. Instead, you passed a DataFrame with shape `{self.predictors.shape}` and a series with shape `{self.target.size}`"
            )

        if (
            self.predictors.select_dtypes(include=np.number).shape[1]
            != self.predictors.shape[1]
        ):
            raise ValueError(
                f"aequilibrium only processes numeric predictors, please pre-process non-numeric predictors on your own and pass aequilibrium only numeric predictors. \n\nNon-numeric columns: {self.predictors.select_dtypes(exclude=np.number).columns}.\n\nIf you want to just use numeric columns already present, you can use the following code: predictors.select_dtypes(include=np.number)"
            )

        if self.predictors.isna().any(axis=1).any():
            raise ValueError(
                f"aequilibrium does not handle NULL values - many balancing functions throw an error with NULLS. Please remove null values from predictors before passing to aequilibrium."
            )

        if self.target.isna().any():
            raise ValueError(
                f"aequilibrium does not handle NULL values - many balancing functions throw an error with NULLS. Please remove null values from target variables before passing to aequilibrium."
            )

        # Populate default values
        self.target_distribution = self.target.value_counts()
        self.num_samples, self.num_predictors = self.predictors.shape
        self.balanced = self.is_balanced()

    def is_balanced(self) -> bool:
        """Checks the balance of this dataset

        Returns:
            bool: True if the dataset is balanced, False otherwise
        """
        ratio = self.target_distribution.max() / self.num_samples

        return bool(ratio <= 0.75 and ratio >= 0.25)
