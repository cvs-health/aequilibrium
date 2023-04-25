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

from typing import Any, Dict

import pandas as pd
from imblearn.metrics import specificity_score
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


class Results:
    def __init__(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        probabilities: pd.Series,
    ):
        """Helper class to manage and evaluate model output

        Args:
            actuals (pd.Series): The actual observed response variable
            predictions (pd.Series): The predicted response variable
            probabilities (pd.Series): The probability of the positive class

        Returns:
            None
        """
        self.actuals = actuals
        self.predictions = predictions
        self.probabilities = probabilities

        if not (
            (self.actuals.size == self.predictions.size)
            and (self.actuals.size == self.probabilities.size)
        ):
            raise ValueError(
                f"All Pandas Series provided to the `Results` class must be the same size."
            )

    @property
    def actuals(self) -> pd.Series:
        return self._actuals

    @actuals.setter
    def actuals(self, new_actuals: Any) -> None:
        if isinstance(new_actuals, pd.Series) and not new_actuals.empty:
            self._actuals = new_actuals.reset_index(drop=True)
        else:
            raise ValueError(
                f"Class `Results` requires parameter `actuals` to be a populated Pandas Series. Instead, you provided an object of type {type(new_actuals)}."
            )

    @property
    def predictions(self) -> pd.Series:
        return self._predictions

    @predictions.setter
    def predictions(self, new_predictions: Any) -> None:
        if isinstance(new_predictions, pd.Series) and not new_predictions.empty:
            self._predictions = new_predictions.reset_index(drop=True)
        else:
            raise ValueError(
                f"Class `Results` requires parameter `predictions` to be a populated Pandas Series. Instead, you provided an object of type {type(new_predictions)}."
            )

    @property
    def probabilities(self) -> pd.Series:
        return self._probabilities

    @probabilities.setter
    def probabilities(self, new_probabilities: Any) -> None:
        if isinstance(new_probabilities, pd.Series) and not new_probabilities.empty:
            self._probabilities = new_probabilities.reset_index(drop=True)
        else:
            raise ValueError(
                f"Class `Results` requires parameter `probabilities` to be a populated Pandas Series. Instead, you provided an object of type {type(new_probabilities)}."
            )

    def to_dict(self) -> Dict[str, pd.Series]:
        return {
            "actuals": self.actuals,
            "predictions": self.predictions,
            "probabilities": self.probabilities,
        }

    def get_true_positives(self) -> int:
        # Both are equal to one
        return (self.actuals.eq(1) & self.predictions.eq(1)).sum()

    def get_true_negatives(self) -> int:
        # Both are equal to zero
        return (self.actuals.eq(0) & self.predictions.eq(0)).sum()

    def get_false_positives(self) -> int:
        # Incorrectly predict one
        return (self.actuals.eq(0) & self.predictions.eq(1)).sum()

    def get_false_negatives(self) -> int:
        # Incorrectly predict zero
        return (self.actuals.eq(1) & self.predictions.eq(0)).sum()

    def get_precision(self) -> float:
        return precision_score(self.actuals, self.predictions)

    def get_recall(self) -> float:
        return recall_score(self.actuals, self.predictions)

    def get_sensitivity(self) -> float:
        return self.get_recall()

    def get_specificity(self) -> float:
        return specificity_score(self.actuals, self.predictions)

    def get_f1_score(self) -> float:
        return f1_score(self.actuals, self.predictions)

    def get_auc_pr_curve(self) -> float:
        precision, recall, thresholds = precision_recall_curve(
            self.actuals, self.probabilities
        )
        return auc(recall, precision)

    def get_confusion_matrix(self) -> pd.DataFrame:
        """Get the confusion matrix for this model results

        Returns:
            pd.DataFrame: Confusion matrix
        """

        confusion_matrix_data = pd.DataFrame(
            confusion_matrix(self.actuals, self.predictions)
        )
        confusion_matrix_data.columns = ["Neg", "Pos"]
        confusion_matrix_data.index = ["Neg", "Pos"]

        return confusion_matrix_data

    def get_summary_table(self, num_bins: int = 5) -> pd.DataFrame:
        """Replacement for enrichment table function

        Args:
            num_bins (int): _description_. Defaults to 5.

        Returns:
            pd.DataFrame: _description_
        """
        # Organize original results
        original_results = pd.DataFrame(self.to_dict())

        # Partition the results based on predicted probabilities
        original_results["bins"] = pd.cut(original_results["probabilities"], num_bins)

        # Aggregate the results
        results = original_results.groupby("bins").agg(
            {
                "probabilities": ["count", "min", "max"],
                "predictions": ["sum"],
                "actuals": ["sum"],
            }
        )

        # Clean up the columns
        results.columns = [
            "num_samples",
            "min_probability",
            "max_probability",
            "num_positive_predictions",
            "num_positive_actuals",
        ]

        return results.sort_index()

    def build_enrich_table(
        self,
        num_decimals: int = 2,
    ) -> pd.DataFrame:
        """Creates enrichment table to be returned and used in lift and gain functions

        Returns:
            pd.DataFrame of enrichment table
        """
        metrics = (
            pd.DataFrame(self.to_dict())
            .sort_values(by="probabilities", ascending=False)
            .reset_index()
        )
        num_rows = metrics.shape[0]
        metrics["percentile"] = (
            pd.Series(range(num_rows)).divide(num_rows).round(num_decimals)
        )

        enrich_df = (
            metrics.groupby("percentile")
            .agg({"probabilities": ["min", "max"], "index": "count", "actuals": "sum"})
            .reset_index()
        )
        enrich_df.columns = [
            "percentile",
            "min_y_proba",
            "max_y_proba",
            "row_count",
            "pos_count",
        ]

        total_events = enrich_df["pos_count"].sum()
        enrich_df["perc_random_events"] = (1 / len(enrich_df["percentile"])) * 100
        # enrich_df[["row_count", "pos_count", "perc_random_events"]] = pd.DataFrame.cumsum(enrich_df)[["row_count", "pos_count", "perc_random_events"]]
        enrich_df[["row_count", "pos_count", "perc_random_events"]] = enrich_df[
            ["row_count", "pos_count", "perc_random_events"]
        ].cumsum()

        enrich_df = enrich_df.assign(
            perc_actual_events=(enrich_df["pos_count"] / total_events) * 100,
            percentile_x_100=enrich_df["percentile"] * 100,
            Random_Lift=1,
            precision=enrich_df["pos_count"] / enrich_df["row_count"],
        )

        return enrich_df.assign(
            Model_Lift=enrich_df["perc_actual_events"]
            / enrich_df["perc_random_events"],
            recall=enrich_df["perc_actual_events"],
        )
