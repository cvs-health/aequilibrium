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

from typing import Any, Dict, List, Optional, Union

import pandas as pd

from aequilibrium.balance import Balance
from aequilibrium.dataset import DataSet
from aequilibrium.model import Model
from aequilibrium.results import Results
from aequilibrium.utils import random_state_generator


def train_models(
    train_dataset: DataSet,
    models: List[Any],
    test_dataset: Optional[DataSet] = None,
    balance_techniques: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, List[Union[Model, Results]]]]:
    """Train models against the dataset using every specified balancing technique

    Args:
        train_dataset (DataSet): Data to use when training models
        models (List[Any]): Classifiers to train against the dataset
        test_dataset (Optional[DataSet]): Data to use when testing models
        balance_techniques (Optional[List[str]]): Function names from the Balance class. Defaults to all techniques.
        random_state (Optional[int]): May provide a random state for reproducibility. Defaults to None.

    Returns:
        Dict: Top level key is a string of the Balance technique, value is a Dictionary.
              Nested Dictionary has three string keys -  `trained_models`, `train_results`, and `test_results`.
              The key `trained_models` is a list of trained classifiers in the form of an aequilibrium `Model` instance.
              The key `train_results` is a list of aequilibrium `Results` instances that capture each model's performance metrics on the train data.
              The key `test_results` is a list of aequilibrium `Results` instances that capture each model's performance metrics on the test data. Only exists if test data is provided.
    """

    # If we don't have an aequilibrium dataset for training
    if not isinstance(train_dataset, DataSet):
        raise ValueError(
            "Please provide an instance of the aequilibrium `DataSet` class for train data."
        )

    # If we don't have an aequilibrium dataset for testing
    if test_dataset is not None and not isinstance(test_dataset, DataSet):
        raise ValueError(
            "Please provide an instance of the aequilibrium `DataSet` class for test data. Or provide nothing."
        )

    # If we don't have a list of classifiers
    if (not isinstance(models, list)) or len(models) == 0:
        raise ValueError(
            "Must provide a populated list of untrained classifier models."
        )

    available_balance_techniques: List[str] = [
        "prototype_generation",
        "random_oversampling",
        "random_undersampling",
        "smote_oversampling",
        "smote_tomek",
        "smote_enn",
        "adasyn_sample",
    ]

    # If not provided, default to every balance technique
    if balance_techniques is None:
        balance_techniques = available_balance_techniques

    # Otherwise, confirm that the provided balance techniques are available
    elif (
        (not isinstance(balance_techniques, list))
        or len(balance_techniques) == 0
        or any(
            balance_technique not in available_balance_techniques
            for balance_technique in balance_techniques
        )
    ):
        raise ValueError(
            f"Must provide a list that is a subset of the available balance techniques: {available_balance_techniques}"
        )

    # Generate a random state for reproducibility
    random_state = random_state_generator(random_state=random_state)

    # Prepare to balance the data
    balance = Balance(dataset=train_dataset, random_state=random_state)

    # Capture relevant output that must be returned
    balanced_dict = dict()

    # Iterate through each balancing technique that must be applied
    for balance_technique in balance_techniques:

        # Initialize tracking of models and results for this balancing technique
        balanced_dict[balance_technique] = dict(
            trained_models=list(), train_results=list()
        )
        if test_dataset is not None:
            balanced_dict[balance_technique]["test_results"] = list()

        # Copy the provided models for training independence
        cur_models = models.copy()

        # Balance the data
        balance_function = getattr(balance, balance_technique)
        cur_train_dataset = balance_function()

        # Iterate through classifiers
        for model in cur_models:

            # Train the model
            cur_model = Model(model)
            train_results: Results = cur_model.fit(cur_train_dataset)

            # Capture the trained model and performance on training data
            balanced_dict[balance_technique]["trained_models"].append(cur_model)
            balanced_dict[balance_technique]["train_results"].append(train_results)

            # Apply the model if necessary
            if test_dataset is not None:
                test_results: Results = cur_model.predict(test_dataset)
                balanced_dict[balance_technique]["test_results"].append(test_results)

    data_tracker = list()

    for balance_technique, value in balanced_dict.items():

        cur_data = dict(balance_technique=balance_technique)

        for model, train_result, test_result in zip(
            value["trained_models"], value["train_results"], value["test_results"]
        ):

            cur_data["model"] = model
            cur_data["train_result"] = train_result
            cur_data["test_result"] = test_result

            data_tracker.append(cur_data.copy())

    # Columns: balance_technique, model, train_result, test_result
    data = pd.DataFrame(data_tracker)
    # Return all the trained models and their performance against the training data
    return data


def get_best_results(
    test_results: List[Results],
    metric: Optional[Union[List[str], str]] = None,
) -> pd.DataFrame:
    """Evaluate models results and rank them from best to worst. 1 is best.

    Note:
        The following metrics are avaliable for model ranking:
        true_positives, true_negatives, false_positives, false_negatives,
        precision, recall, sensitivity, specificity, f1_score

    Args:
        test_results (List[Results]): Multiple results instances from a variety of models
        metric (Union[List[str], str]): A single metric or multiple metrics to consider when ranking

    Returns:
        pd.DataFrame: The performance metrics for each results instance.
                      Including a `rank` column where 0 indicates the best results.
    """

    # Initialize dictionary to track metrics
    evaluation_metrics = dict(
        true_positives=list(),
        true_negatives=list(),
        false_positives=list(),
        false_negatives=list(),
        precision=list(),
        recall=list(),
        sensitivity=list(),
        specificity=list(),
        f1_score=list(),
        auc_pr=list(),
    )

    for result in test_results:
        evaluation_metrics["true_positives"].append(result.get_true_positives())
        evaluation_metrics["true_negatives"].append(result.get_true_negatives())
        evaluation_metrics["false_positives"].append(result.get_false_positives())
        evaluation_metrics["false_negatives"].append(result.get_false_negatives())
        evaluation_metrics["precision"].append(result.get_precision())
        evaluation_metrics["recall"].append(result.get_recall())
        evaluation_metrics["sensitivity"].append(result.get_sensitivity())
        evaluation_metrics["specificity"].append(result.get_specificity())
        evaluation_metrics["f1_score"].append(result.get_f1_score())
        evaluation_metrics["auc_pr"].append(result.get_auc_pr_curve())

    available_metrics: List[str] = [
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1_score",
        "auc_pr",
    ]

    # If not provided, default to every metric
    if metric is None:
        metric = available_metrics

    # Rank the results
    evaluation_data = pd.DataFrame(evaluation_metrics)
    evaluation_data["rank"] = pd.Series(
        evaluation_data[metric].itertuples(index=False)
    ).rank(axis=0, method="min", ascending=False)

    return evaluation_data


def train_and_evaluate(
    train_dataset: DataSet,
    test_dataset: DataSet,
    models: List[Any],
    metric: Optional[Union[List[str], str]] = None,
    sort_metric: Optional[str] = "auc_pr",
    balance_techniques: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Any:
    """Train models against the dataset using every specified balancing technique

    Note:
        The following metrics are avaliable for model ranking:
        true_positives, true_negatives, false_positives, false_negatives,
        precision, recall, sensitivity, specificity, f1_score

    Args:
        train_dataset (DataSet): Data to use when training models
        test_dataset (DataSet): Data to use when testing models
        models (List[Any]): Classifiers to train against the dataset
        metric (Union[List[str], str]): A single metric or multiple metrics to consider when ranking
        balance_techniques (Optional[List[str]]): Function names from the Balance class. Defaults to all techniques.
        random_state (Optional[int]): May provide a random state for reproducibility. Defaults to None.

    Returns:

    """

    # Train models and get results
    data = train_models(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        models=models,
        balance_techniques=balance_techniques,
        random_state=random_state,
    )

    # Rank the test results based on some metric
    result_ranking = get_best_results(
        test_results=data["test_result"].tolist(), metric=metric
    )

    # Combine the training results and the evaluation ranking metrics
    return (
        pd.concat([data, result_ranking], axis=1)
        .sort_values(by=sort_metric, ascending=False)
        .reset_index(drop=True)
    )
