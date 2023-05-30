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

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

from aequilibrium.results import Results


class Visualize:
    def __init__(
        self,
        results: Results,
        num_decimals: int = 2,
    ):
        """My class description

        Args:
            results (Results): An aequilibrium Results instance
            num_decimals (int): Desired rounding for floats

        Returns:
            None
        """
        self.results = results

        self.num_decimals = num_decimals
        self.metric_summary = self.results.build_enrich_table(
            num_decimals=self.num_decimals
        )

    def plot_pr_curve(
        self,
        display: bool = True,
        save_file_name: Optional[str] = None,
    ) -> None:
        """Displays a precision recall curve of the given results of a model

        Args:
            display (bool): True displays PR Curve
            save_file_name (str): Filename to save plot to file

        Returns:
            TODO
        """
        # Data to plot precision - recall curve
        precision, recall, _ = precision_recall_curve(
            self.results.actuals, self.results.probabilities
        )

        pr_curve = PrecisionRecallDisplay(precision=precision, recall=recall)

        pr_curve.plot(color="blue")
        plt.title("Precision-Recall Curve", fontsize=28, fontweight="bold")
        plt.xlabel("Recall", fontsize=22)
        plt.ylabel("Precision", fontsize=22)

        if save_file_name:
            plt.savefig(save_file_name)

        if display:
            plt.show()

    def plot_confusion_matrix(
        self,
        display: bool = True,
        save_file_name: Optional[str] = None,
    ) -> None:
        """Function that displays a heatmap of a confusion matrix.

        Args:
            display (bool): True displays PR Curve
            save_file_name (str): Filename to save plot to file

        Returns:
            None
        """
        confusion_matrix_data = self.results.get_confusion_matrix()

        sb.heatmap(confusion_matrix_data, annot=True, fmt="g", cmap="viridis")
        plt.title("Confusion Matrix", fontsize=28, fontweight="bold")
        plt.xlabel("Predicted", fontsize=22)
        plt.ylabel("Actual", fontsize=22)

        if save_file_name:
            plt.savefig(save_file_name)

        if display:
            plt.show()

    def plot_gain_chart(
        self,
        display: bool = True,
        save_file_name: Optional[str] = None,
    ) -> None:
        """Displays a gain chart of the given results of a model

        Args:
            display (bool): True displays gain chart
            save_file_name (str): Filename to save plot to file

        Returns:
            None
        """

        enrich_table = self.metric_summary

        dict = {}
        for keys in enrich_table:
            dict[keys] = 0.0

        new_row = pd.DataFrame(dict, index=[0])
        gain_df = pd.concat([new_row, enrich_table]).reset_index(drop=True)

        sb.lineplot(
            x="percentile_x_100",
            y="perc_random_events",
            data=gain_df,
            dashes=False,
            color="red",
            marker="o",
            label="Random",
            sort=False,
        )

        sb.lineplot(
            x="percentile_x_100",
            y="perc_actual_events",
            data=gain_df,
            dashes=False,
            color="blue",
            marker="o",
            label="Model",
            sort=False,
        )

        plt.title("Gain Chart", fontsize=32, fontweight="bold")
        plt.ylabel("Gain", fontsize=22)
        plt.xlabel("% of Dataset", fontsize=22)
        plt.grid(axis="x", color="0.95")
        plt.grid(axis="y", color="0.95")
        plt.xticks(np.arange(0, 101, step=5))
        plt.yticks(np.arange(0, 101, step=5))

        if save_file_name:
            plt.savefig(save_file_name)

        if display:
            plt.show()

    def plot_lift_chart(
        self,
        display: bool = True,
        save_file_name: Optional[str] = None,
    ) -> None:
        """Displays a lift chart of the given results of a model

        Args:
            display (bool): True displays gain chart
            save_file_name (str): Filename to save plot to file

        Returns:
            None
        """

        enrich_table = self.metric_summary
        sb.lineplot(
            x="percentile_x_100",
            y="Random_Lift",
            data=enrich_table,
            dashes=False,
            color="red",
            marker="o",
            label="Random_Lift",
            sort=False,
        )
        sb.lineplot(
            x="percentile_x_100",
            y="Model_Lift",
            data=enrich_table,
            dashes=False,
            color="blue",
            marker="o",
            label="Model_Lift",
            sort=False,
        )

        plt.title("Lift Chart", fontsize=32, fontweight="bold")
        plt.xlabel("Percent Group", fontsize=22)
        plt.ylabel("Lift", fontsize=22)
        plt.grid(axis="x", color="0.95")
        plt.grid(axis="y", color="0.95")
        plt.xticks(np.arange(0, 100, step=5))

        if save_file_name:
            plt.savefig(save_file_name)

        if display:
            plt.show()

    def complete_evaluation(
        self,
        display: bool = True,
        save_dir: Optional[str] = False,
    ) -> pd.DataFrame:
        """Creates evaluation metrics from results dataframe

        Args:
            display (bool): True displays gain chart
            save_dir (str): File directory to save plots

        Returns:
            Enrichment dataframe
        """

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            else:
                print(
                    f"WARNING: {save_dir} directory already existed. Files may have been overwritten."
                )
            self.plot_pr_curve(display, f"{os.path.join(save_dir, 'pr_curve')}")
            self.plot_confusion_matrix(
                display, f"{os.path.join(save_dir, 'conf_matrix')}"
            )
            self.plot_gain_chart(display, f"{os.path.join(save_dir, 'gain_chart')}")
            self.plot_lift_chart(display, f"{os.path.join(save_dir, 'lift_chart')}")
        else:
            self.plot_pr_curve(display)
            self.plot_confusion_matrix(display)
            self.plot_gain_chart(display)
            self.plot_lift_chart(display)

        return self.metric_summary
