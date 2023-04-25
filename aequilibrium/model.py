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

from typing import Any, Tuple

import pandas as pd

from aequilibrium.dataset import DataSet
from aequilibrium.results import Results


class Model:
    def __init__(
        self,
        model: Any,
    ):
        """Helper class to train a model

        Args:
            model (Any): A model to consider

        Returns:
            None
        """
        self.model = model

    @property
    def model(self) -> Any:
        return self._model

    @model.setter
    def model(self, new_model: Any) -> None:
        if (
            hasattr(new_model, "fit")
            and hasattr(new_model, "predict")
            and hasattr(new_model, "predict_proba")
        ):
            self._model = new_model
        else:
            raise ValueError(
                f"Class `Model` requires parameter `model` to have the attributes `fit` and `predict` and `predict_proba` available."
            )

    def fit(self, train_data: DataSet) -> Results:
        """Train a model against this dataset

        Args:
            train_data (DataSet): A valid instance of an aequilibrium `DataSet`

        Returns:
            Results: Metrics for this model applied to the training data
        """
        if not train_data.balanced:
            raise ValueError("Please balance your training data.")

        self.model.fit(train_data.predictors, train_data.target)

        # Train Data Stats
        train_actuals = pd.Series(train_data.target)
        train_predictions = pd.Series(self.model.predict(train_data.predictors))
        train_probabilities = pd.Series(
            self.model.predict_proba(train_data.predictors)[:, 1]
        )

        return Results(
            actuals=train_actuals,
            predictions=train_predictions,
            probabilities=train_probabilities,
        )

    def predict(self, test_data: DataSet) -> Results:
        """Calculate prediction stats against this dataset

        Args:
            test_data (DataSet): A valid instance of an aequilibrium `DataSet`

        Returns:
            Results: Metrics for this model applied to the test data
        """
        if test_data.balanced:
            raise ValueError(
                "Do not balance your test data! aequilibrium is only useful for imbalanced data."
            )

        # Test Data Stats
        test_actuals = pd.Series(test_data.target)
        test_predictions = pd.Series(self.model.predict(test_data.predictors))
        test_probabilities = pd.Series(
            self.model.predict_proba(test_data.predictors)[:, 1]
        )

        return Results(
            actuals=test_actuals,
            predictions=test_predictions,
            probabilities=test_probabilities,
        )

    def fit_predict(
        self, train_data: DataSet, test_data: DataSet
    ) -> Tuple[Results, Results]:
        """Train a model against this dataset and capture predictions

        Args:
            train_data (DataSet): A valid instance of an aequilibrium `DataSet`
            test_data (DataSet): A valid instance of an aequilibrium `DataSet`

        Returns:
            Results: Metrics for this model applied to the training data
            Results: Metrics for this model applied to the test data
        """
        # Fit a model
        train_results = self.fit(train_data)

        # Apply the model
        test_results = self.predict(test_data)

        return train_results, test_results
