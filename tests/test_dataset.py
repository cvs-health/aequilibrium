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

import unittest

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from aequilibrium.dataset import DataSet


class TestCompute(unittest.TestCase):
    hypothesis_dataset = data_frames(
        index=range_indexes(min_size=4),
        columns=[
            column(
                "A",
                dtype=np.int64,
                elements=st.integers(
                    min_value=np.iinfo(np.int64).min, max_value=np.iinfo(np.int64).max
                ),
            ),
            column(
                "B",
                dtype=np.int64,
                elements=st.integers(
                    min_value=np.iinfo(np.int64).min, max_value=np.iinfo(np.int64).max
                ),
            ),
            column(
                "C",
                dtype=np.float64,
                elements=st.floats(
                    min_value=np.finfo(np.float64).min,
                    max_value=np.finfo(np.float64).max,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            column(
                "D",
                dtype=np.float64,
                elements=st.floats(
                    min_value=np.finfo(np.float64).min,
                    max_value=np.finfo(np.float64).max,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            column("target", dtype=bool, elements=st.booleans()),
        ],
    )

    # This function runs once before all the tests in this file
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.target_column = "target"

    # This function runs once prior to each test
    def setUp(self, *args, **kwargs) -> None:
        self.predictors = pd.DataFrame(
            {
                "A": list(range(10)),
                "B": list(range(5)) * 2,
                "C": [i / 10 for i in range(10)],
                "D": [i / 20 for i in range(10)],
            }
        )
        self.target = pd.Series([False, True] * 5)

        return super().setUp(*args, **kwargs)

    def _fix_hypothesis_dataset(self, dataset: pd.DataFrame):
        dataset.at[0, "target"] = False
        dataset.at[1, "target"] = True
        return dataset

    @given(hypothesis_dataset)
    def test_hypothesis_initialization(self, dataset: pd.DataFrame):
        dataset = self._fix_hypothesis_dataset(dataset)
        predictors = dataset.drop(columns=[self.target_column])
        target = dataset[self.target_column]
        new_dataset = DataSet(predictors=predictors, target=target)

        self.assertTrue(predictors.equals(new_dataset.predictors))
        self.assertTrue(target.equals(new_dataset.target))
        self.assertTrue(target.value_counts().equals(new_dataset.target_distribution))
        self.assertEqual(new_dataset.num_predictors, 4)
        self.assertEqual(new_dataset.num_samples, target.size)

    @given(hypothesis_dataset)
    def test_hypothesis_property_existence(self, dataset: pd.DataFrame):
        dataset = self._fix_hypothesis_dataset(dataset)
        predictors = dataset.drop(columns=[self.target_column])
        target = dataset[self.target_column]
        new_dataset = DataSet(predictors=predictors, target=target)

        self.assertTrue(hasattr(new_dataset, "predictors"))
        self.assertTrue(hasattr(new_dataset, "target"))
        self.assertTrue(hasattr(new_dataset, "target_distribution"))
        self.assertTrue(hasattr(new_dataset, "num_predictors"))
        self.assertTrue(hasattr(new_dataset, "num_samples"))
        self.assertTrue(hasattr(new_dataset, "balanced"))

    @given(hypothesis_dataset)
    def test_hypothesis_property_dtype(self, dataset: pd.DataFrame):
        dataset = self._fix_hypothesis_dataset(dataset)
        predictors = dataset.drop(columns=[self.target_column])
        target = dataset[self.target_column]
        new_dataset = DataSet(predictors=predictors, target=target)

        self.assertIsInstance(new_dataset.predictors, pd.DataFrame)
        self.assertIsInstance(new_dataset.target, pd.Series)
        self.assertIsInstance(new_dataset.target_distribution, pd.Series)
        self.assertIsInstance(new_dataset.num_predictors, int)
        self.assertIsInstance(new_dataset.num_samples, int)
        self.assertIsInstance(new_dataset.balanced, bool)

    @given(hypothesis_dataset)
    def test_hypothesis_balance(self, dataset: pd.DataFrame):
        dataset = self._fix_hypothesis_dataset(dataset)
        predictors = dataset.drop(columns=[self.target_column])
        target = dataset[self.target_column]
        new_dataset = DataSet(predictors=predictors, target=target)

        ratio = new_dataset.target_distribution.max() / new_dataset.num_samples
        expected_balance = bool(ratio <= 0.75 and ratio >= 0.25)

        self.assertEqual(expected_balance, new_dataset.balanced)

    def test_valid_initialization(self):
        new_dataset = DataSet(predictors=self.predictors, target=self.target)

        self.assertTrue(self.predictors.equals(new_dataset.predictors))
        self.assertTrue(self.target.equals(new_dataset.target))

    def test_unequal_size(self):
        with self.assertRaises(ValueError):
            _ = DataSet(predictors=self.predictors.loc[:5], target=self.target)

        with self.assertRaises(ValueError):
            _ = DataSet(predictors=self.predictors, target=self.target.loc[:5])

    def test_binary_target(self):
        target = pd.Series([0] * self.target.size)
        with self.assertRaises(ValueError):
            _ = DataSet(predictors=self.predictors, target=target)

        target = pd.Series(list(range(self.target.size)))
        with self.assertRaises(ValueError):
            _ = DataSet(predictors=self.predictors, target=target)

    def test_invalid_parameter_types(self):

        with self.assertRaises(TypeError):
            _ = DataSet(predictors="hello", target=self.target)

        with self.assertRaises(TypeError):
            _ = DataSet(predictors=pd.Series(["world"]), target=self.target)

        with self.assertRaises(TypeError):
            _ = DataSet(predictors=123, target=self.target)

        with self.assertRaises(TypeError):
            _ = DataSet(predictors=self.predictors, target="hello")

        with self.assertRaises(TypeError):
            _ = DataSet(predictors=self.predictors, target=pd.DataFrame(["world"]))

        with self.assertRaises(TypeError):
            _ = DataSet(predictors=self.predictors, target=123)

    def test_invalid_init(self):

        with self.assertRaises(TypeError):
            _ = DataSet(target=self.target)

        with self.assertRaises(TypeError):
            _ = DataSet(predictors=self.predictors)

        with self.assertRaises(TypeError):
            _ = DataSet(
                predictors=self.predictors, target=self.target, target_distribution=None
            )

        with self.assertRaises(TypeError):
            _ = DataSet(predictors=self.predictors, target=self.target, balanced=False)


if __name__ == "__main__":
    unittest.main()
