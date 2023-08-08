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

from typing import Any, Dict, Optional

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

from aequilibrium.dataset import DataSet
from aequilibrium.utils import random_state_generator


class Balance:
    def __init__(
        self,
        dataset: DataSet,
        random_state: Optional[int] = None,
    ):
        """Identify imbalance and resolve imbalance within datasets

        Args:
            dataset (DataSet): A valid aequilibrium DataSet
            random_state (Optional[int]): Enable reproducibility by specifying random state

        Returns:
            None
        """

        self.dataset = dataset
        self.random_state = random_state

    @property
    def dataset(self) -> DataSet:
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset: DataSet) -> None:
        if isinstance(new_dataset, DataSet):
            self._dataset = new_dataset
        else:
            raise ValueError(
                f"Must provide a valid instance of the aequilibrium `DataSet` class. Instead, you provided an object with type `{type(new_dataset)}`."
            )

    @property
    def random_state(self) -> int:
        if self._random_state is None:
            return random_state_generator()
        else:
            return self._random_state

    @random_state.setter
    def random_state(self, new_random_state: int) -> None:
        if (new_random_state is None) or isinstance(new_random_state, int):
            self._random_state = new_random_state
        else:
            raise ValueError(
                f"Must provide either `None` or an integer for the `Balance` class parameter `random_state`. Instead you have provided and object of type `{type(new_random_state)}`"
            )

    def prototype_generation(self) -> DataSet:
        """Transforms imbalanced data by undersampling the majority class and
           generating new data entries of the minority class.

        Returns:
            DataSet: A new balanced dataset
        """
        undersampler = ClusterCentroids()
        predictors, target = undersampler.fit_resample(
            self.dataset.predictors, self.dataset.target
        )
        return DataSet(predictors, target)

    def random_oversampling(self, ratio: Optional[float] = None) -> DataSet:
        """Performs random over sampling of minority class

        Args:
            ratio (Optional[float]): TODO

        Returns:
            DataSet: A new balanced dataset
        """
        os = RandomOverSampler(random_state=self.random_state)
        predictors, target = os.fit_resample(
            self.dataset.predictors, self.dataset.target
        )
        return DataSet(predictors, target)

    def random_undersampling(self, ratio: Optional[float] = None) -> DataSet:
        """Helper function that performs random under sampling of majority class

        Args:
            ratio (Optional[float]): TODO

        Returns:
            DataSet: A new balanced dataset
        """
        os = RandomUnderSampler(random_state=self.random_state)
        predictors, target = os.fit_resample(
            self.dataset.predictors, self.dataset.target
        )
        return DataSet(predictors, target)

    def smote_oversampling(self) -> DataSet:
        """Uses base SMOTE to preprocess imbalanced data

        Returns:
            DataSet: A new balanced dataset
        """
        smote = SMOTE(random_state=self.random_state)
        predictors, target = smote.fit_resample(
            self.dataset.predictors, self.dataset.target
        )
        return DataSet(predictors, target)

    def smote_tomek(self) -> DataSet:
        """Helper function that combines SMOTE and Tomek Links to balance imbalanaced data

        Returns:
            DataSet: A new balanced dataset
        """
        smote = SMOTETomek(random_state=self.random_state)
        predictors, target = smote.fit_resample(
            self.dataset.predictors, self.dataset.target
        )
        return DataSet(predictors, target)

    def smote_enn(self) -> DataSet:
        """Helper function that uses SMOTE and Edited Nearest Neighbors to
        combine over and under sampling to balance imbalanced data.

        Returns:
            DataSet: A new balanced dataset
        """
        smote = SMOTEENN(random_state=self.random_state)
        predictors, target = smote.fit_resample(
            self.dataset.predictors, self.dataset.target
        )
        return DataSet(predictors, target)

    def adasyn_sample(self) -> DataSet:
        """Helper function that uses ADASYN's hybrid approach to data
        balancing to balance imbalanced data

        Returns:
            DataSet: A new balanced dataset
        """
        ada = ADASYN(random_state=self.random_state)
        predictors, target = ada.fit_resample(
            self.dataset.predictors, self.dataset.target
        )
        return DataSet(predictors, target)

    def overweight(self) -> Dict[str, float]:
        """TODO

        Returns:
            Dict[str, float]: TODO
        """
        max_val = self.dataset.target_distribution.max()
        return (max_val / self.dataset.target_distribution).to_dict()

    def underweight(self) -> Dict[Any, float]:
        """TODO

        Returns:
            Dict[str, float]: TODO
        """
        min_val = self.dataset.target_distribution.min()
        return (min_val / self.dataset.target_distribution).to_dict()

    def balance_data(self) -> Dict[str, DataSet]:
        """Run all modes of balancing data

        Returns:
            Dict[str, float]: A Pandas DataFrame for each balancing technique
        """
        return {
            "random_oversampling": self.random_oversampling(),
            "random_undersampling": self.random_undersampling(),
            # "prototype_generation": self.prototype_generation(),
            "smote_oversampling": self.smote_oversampling(),
            # "smote_tomek":self.smote_tomek(),
            # "smote_enn":self.smote_enn(),
            # "adasyn_sample":self.adasyn_sample()
        }
