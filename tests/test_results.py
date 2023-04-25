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

import pandas as pd
from hypothesis import given
from hypothesis.extra.pandas import column, data_frames, range_indexes

from aequilibrium.results import Results


class TestResults(unittest.TestCase):
    hypothesis_results = data_frames(
        index=range_indexes(min_size=1),
        columns=[
            column(
                "y_true",
                dtype=int,
            ),
            column("y_pred", dtype=int),
            column("y_proba", dtype=float),
        ],
    )

    # This function runs once before all the tests in this file
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.target_column = "target"

    # This function runs once prior to each test
    def setUp(self, *args, **kwargs) -> None:
        self.result_dataframe = pd.DataFrame(
            {
                "y_true": [True, False, True] * 4,
                "y_pred": [True, False, False] * 4,
                "y_proba": [i / 10 for i in range(12)],
            }
        )

        return super().setUp(*args, **kwargs)

    @given(hypothesis_results)
    def test_hypothesis_initialization(self, dataset: pd.DataFrame):
        actuals = dataset["y_true"]
        predictions = dataset["y_pred"]
        probabilities = dataset["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertTrue(actuals.equals(new_results.actuals))
        self.assertTrue(predictions.equals(new_results.predictions))
        self.assertTrue(probabilities.equals(new_results.probabilities))

    @given(hypothesis_results)
    def test_hypothesis_property_existence(self, dataset: pd.DataFrame):
        actuals = dataset["y_true"]
        predictions = dataset["y_pred"]
        probabilities = dataset["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertTrue(hasattr(new_results, "actuals"))
        self.assertTrue(hasattr(new_results, "predictions"))
        self.assertTrue(hasattr(new_results, "probabilities"))

    @given(hypothesis_results)
    def test_hypothesis_property_dtype(self, dataset: pd.DataFrame):
        actuals = dataset["y_true"]
        predictions = dataset["y_pred"]
        probabilities = dataset["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertIsInstance(new_results.actuals, pd.Series)
        self.assertIsInstance(new_results.predictions, pd.Series)
        self.assertIsInstance(new_results.probabilities, pd.Series)

    def test_valid_initialization(self):

        new_results = Results(
            actuals=self.result_dataframe.y_true,
            predictions=self.result_dataframe.y_pred,
            probabilities=self.result_dataframe.y_proba,
        )

        self.assertTrue(self.result_dataframe.y_true.equals(new_results.actuals))
        self.assertTrue(self.result_dataframe.y_pred.equals(new_results.predictions))
        self.assertTrue(self.result_dataframe.y_proba.equals(new_results.probabilities))

    def test_unequal_size(self):
        with self.assertRaises(ValueError):
            _ = Results(
                actuals=self.result_dataframe.y_true.loc[:5],
                predictions=self.result_dataframe.y_pred,
                probabilities=self.result_dataframe.y_proba,
            )

        with self.assertRaises(ValueError):
            _ = Results(
                actuals=self.result_dataframe.y_true,
                predictions=self.result_dataframe.y_pred.loc[:5],
                probabilities=self.result_dataframe.y_proba,
            )

        with self.assertRaises(ValueError):
            _ = Results(
                actuals=self.result_dataframe.y_true,
                predictions=self.result_dataframe.y_pred,
                probabilities=self.result_dataframe.y_proba.loc[:5],
            )

    def test_truepos(self):
        actuals = self.result_dataframe["y_true"]
        predictions = self.result_dataframe["y_pred"]
        probabilities = self.result_dataframe["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertTrue(new_results.get_true_positives() == 4)

    def test_trueneg(self):
        actuals = self.result_dataframe["y_true"]
        predictions = self.result_dataframe["y_pred"]
        probabilities = self.result_dataframe["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertTrue(new_results.get_true_negatives() == 4)

    def test_falsepos(self):
        actuals = self.result_dataframe["y_true"]
        predictions = self.result_dataframe["y_pred"]
        probabilities = self.result_dataframe["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertTrue(new_results.get_false_positives() == 0)

    def test_falseneg(self):
        actuals = self.result_dataframe["y_true"]
        predictions = self.result_dataframe["y_pred"]
        probabilities = self.result_dataframe["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertTrue(new_results.get_false_negatives() == 4)

    def test_precision(self):
        actuals = self.result_dataframe["y_true"]
        predictions = self.result_dataframe["y_pred"]
        probabilities = self.result_dataframe["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertTrue(new_results.get_precision() == 1.0)

    def test_recall(self):
        actuals = self.result_dataframe["y_true"]
        predictions = self.result_dataframe["y_pred"]
        probabilities = self.result_dataframe["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertTrue(new_results.get_recall() == 0.5)

    def test_sensitivity(self):
        actuals = self.result_dataframe["y_true"]
        predictions = self.result_dataframe["y_pred"]
        probabilities = self.result_dataframe["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertTrue(new_results.get_sensitivity() == 0.5)

    def test_specificity(self):
        actuals = self.result_dataframe["y_true"]
        predictions = self.result_dataframe["y_pred"]
        probabilities = self.result_dataframe["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertTrue(new_results.get_specificity() == 1.0)

    def test_f1score(self):
        actuals = self.result_dataframe["y_true"]
        predictions = self.result_dataframe["y_pred"]
        probabilities = self.result_dataframe["y_proba"]

        new_results = Results(
            actuals=actuals, predictions=predictions, probabilities=probabilities
        )

        self.assertTrue(new_results.get_f1_score() == 2 / 3)


if __name__ == "__main__":
    unittest.main()
