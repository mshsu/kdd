# CSC 466 Fall 2023
# Martin Hsu - mshsu@calpoly.edu
# Rachel Roggenkemper - rroggenk@calpoly.edu

import pandas as pd
import numpy as np
import sys

from recommend import mean_utility, wtd_sum, adj_wtd_sum, \
    similarity_matrix, decision
from typing import Optional, Callable


def eval_random(data: pd.DataFrame, size: int = 200,
                method: str = "user", func: Callable = mean_utility,
                n: Optional[int] = None) -> pd.DataFrame:
    """
    Performs ratings and recommendation predictions on a random sample of user
    and item pairs.

    :param data: A pandas dataframe representing the original ratings database
    :param size: (Default 200) An integer representing sample size
    :param method: (Default "user") A string, either "user" or "item," which
        specifies whether to use user-based or item-based recommendations.
    :param func: (Default mean_utility) Function with which to perform ratings
        predictions
    :param n: (Default None) An optional integer, where if specified will denote
        the number of neighbors with which to calculate the rating.
    :return: A pandas dataframe with the following columns: user, item, observed
        rating, predicted rating, absolute difference in rating between observed
        and predicted rating, observed recommendation (T/F), predicted
        recommendation (T/F)
    """
    # Get a dataframe of valid user-item pairs
    # Columns: user, item, observed rating
    stacked = data.stack() \
        .replace(99, np.nan) \
        .dropna() \
        .sample(size) \
        .reset_index() \
        .rename(columns={"level_0": "user", "level_1": "item", 0: "obs_rating"})
    # We will not predict using the whole dataset; we will just use all possible
    #   pairs of unique item/user found in the sample. This will make it faster
    sample = data.loc[stacked["user"].unique(), stacked["item"].unique()] \
        .sort_index().T \
        .sort_index().T \
        .replace(99, np.nan)

    # Make similarity matrix
    sim = similarity_matrix(sample, method=method)

    # Get predicted rating
    results = stacked.copy()
    results['pred_rating'] = pd.Series(stacked.index) \
        .apply(lambda i: func(sample, sim,
                              stacked.loc[i, "user"], stacked.loc[i, "item"],
                              method=method, n=n))

    # Calculate change in rating, and whether to recommend for both observed and
    #   predicted ratings
    results['delta_rating'] = np.abs(
        results['obs_rating'] - results['pred_rating']
    )
    results['obs_rec'] = results['obs_rating'].apply(decision)
    results['pred_rec'] = results['pred_rating'].apply(decision)

    return results


def eval_report(results: pd.DataFrame) -> None:
    """
    Prints a report for ratings predictions performance.

    :param results: A pandas dataframe output of eval_random
    :return: None
    """
    # Make confusion matrix
    conf_matrix = pd.crosstab(results['pred_rec'], results['obs_rec'])
    conf_matrix = conf_matrix.reindex(index=conf_matrix.columns,
                                      columns=conf_matrix.columns,
                                      fill_value=0) \
        .rename_axis("obs", axis=1).rename_axis("pred", axis=0)

    # Calculate TP, TN, FP, FN
    tp = conf_matrix.iloc[1, 1]
    tn = conf_matrix.iloc[0, 0]
    fp = conf_matrix.iloc[1, 0]
    fn = conf_matrix.iloc[0, 1]

    # Calculate precision, recall, F1, accuracy, MAE
    precision = tp / (tp + fp)
    if (tp + fp == 0) and (tp == 0):
        precision = 1
    recall = tp / (tp + fn)
    if (tp + fn == 0) and (tp == 0):
        recall = 1
    f1 = max(0,  2 * precision * recall / (precision + recall))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    mae = results['delta_rating'].mean()

    # Print report
    print("Confusion Matrix:")
    print(conf_matrix)
    print()
    print(f"Precision:        {precision:0.3f}")
    print(f"Recall:           {recall:0.3f}")
    print(f"F1 Measure:       {f1:0.3f}")
    print(f"Overall Accuracy: {accuracy:0.3f}")
    print(f"MAE:              {mae:0.3f}")
    print()
    # Print entire sample and predictions
    print("Predictions:")
    print(results[['user', 'item', 'obs_rating',
                   'pred_rating', 'delta_rating']].to_string(index=False))


def main():
    args = sys.argv

    # If no parameters, print help message
    if len(args) < 4:
        print("Syntax:  python3 EvaluateCFRandom.py Method Size Repeats")
        print("Methods: user-mean-utility | user-weighted-sum |"
              " user-adjusted-weighted-sum")
        print("         item-mean-utility | item-weighted-sum |"
              " item-adjusted-weighted-sum")

    else:
        # Assign function parameters from system input
        method_whole = args[1]
        method_whole_list = method_whole.split("-", 1)
        method = method_whole_list[0]
        func_str = method_whole_list[1]
        size = int(args[2])
        repeats = int(args[3])
        n = 5

        # Read jester data
        data = pd.read_csv("data/jester-data-1.csv", header=None) \
            .drop(0, axis=1) \
            .T.reset_index(drop=True).T

        # Assign CF algorithm
        func = None
        if func_str == "mean-utility":
            func = mean_utility
        elif func_str == "weighted-sum":
            func = wtd_sum
        elif func_str == "adjusted-weighted-sum":
            func = adj_wtd_sum

        method_whole_name = " ".join(
            [word.capitalize() for word in method_whole.split("-")])

        # Print report
        print("Random Sample Report")
        print(f"Method: {method_whole_name}")
        print(f"Sample Size: {size}")
        print(f"N Neighbors: {n}")
        print()
        for i in range(repeats):
            print("-" * 80)
            print()
            print(f"Repeat: {i}")
            results = eval_random(data, method=method, size=size,
                                  func=func, n=5)
            eval_report(results)
            print()


if __name__ == "__main__":
    main()
