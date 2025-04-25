# CSC 466 Fall 2023
# Martin Hsu - mshsu@calpoly.edu
# Rachel Roggenkemper - rroggenk@calpoly.edu

import pandas as pd
import numpy as np
import sys

from recommend import mean_utility, wtd_sum, adj_wtd_sum, \
    similarity_matrix, decision
from typing import Optional, Callable


def eval_list(data: pd.DataFrame, samp_list: pd.DataFrame,
              method: str = "user", func: Callable = mean_utility,
              n: Optional[int] = None) -> pd.DataFrame:
    """
    Performs ratings and recommendation predictions on a specified list of
    user and item pairs.

    :param data: A pandas dataframe representing the original ratings database
    :param samp_list: A pandas dataframe, with the first column denoting
        the user ID and the second column denoting item ID
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
    stacked = samp_list.copy()
    # Get observed rating for each user-item pair
    stacked['obs_rating'] = pd.Series(stacked.index) \
        .apply(lambda i: data.loc[stacked.loc[i, 'user'],
                                  stacked.loc[i, 'item']]) \
        .replace(99, np.nan)
    # Drop invalid pairs (no observed rating)
    stacked = stacked.dropna().reset_index(drop=True)
    # We will not predict using the whole dataset; we will just use all possible
    #   pairs of unique item/user found in the sample. This will make it faster
    sample = data.loc[stacked["user"].unique(), stacked["item"].unique()] \
        .sort_index().T.sort_index().T.replace(99, np.nan)

    # Make similarity matrix
    sim = similarity_matrix(sample)

    # Get predicted rating
    results = stacked.copy()
    results['pred_rating'] = pd.Series(stacked.index) \
        .apply(lambda i: func(sample, sim,
                              stacked.loc[i, "user"], stacked.loc[i, "item"],
                              method=method, n=n))

    # Calculate change in rating, and whether to recommend for both observed and
    #   predicted ratings
    results['delta_rating'] = np.abs(results['obs_rating']
                                     - results['pred_rating'])
    results['obs_rec'] = results['obs_rating'].apply(decision)
    results['pred_rec'] = results['pred_rating'].apply(decision)

    return results


def eval_report(results: pd.DataFrame) -> None:
    """
    Prints a report for ratings predictions performance.

    :param results: A pandas dataframe output of eval_list
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
    f1 = max(0, 2 * precision * recall / (precision + recall))
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


def generate_sample(out: str, size: int = 200, all_valid: bool = True) -> None:
    """
    Generates random user-item pairs from the jester data. Not used in reporting
    but is included for exploratory and debugging purposes.

    :param out: Location of output
    :param size: Size of sample
    :param all_valid: (Default True) If True, all sampled pairs have a valid
        observed value.
    :return: None
    """
    jester = pd.read_csv("data/jester-data-1.csv", header=None) \
        .drop(0, axis=1).T \
        .reset_index(drop=True).T

    stacked = jester.stack() \
        .replace(99, np.nan)
    if all_valid:
        stacked = stacked.dropna()
    samp_list = stacked.sample(size) \
        .reset_index() \
        .rename(columns={"level_0": "user", "level_1": "item"}) \
        .drop(0, axis=1)
    samp_list.to_csv(out, index=False, header=False)


def main():
    args = sys.argv

    # If no parameters, print help message
    if len(args) < 3:
        print("Syntax:  python3 EvaluateCFRandom.py Method Filename")
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
        filename = args[2]

        # Read jester data
        data = pd.read_csv("data/jester-data-1.csv", header=None) \
            .drop(0, axis=1) \
            .T.reset_index(drop=True).T
        # Read sample pairs
        samp_list = pd.read_csv(filename, header=None)
        samp_list.columns = ['user', 'item']

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
        filename_basic = filename.split("/")[-1]
        n = 5

        # Print report
        print("List Sample Report")
        print(f"Method: {method_whole_name}")
        print(f"Sample File: {filename_basic}")
        print(f"N Neighbors: {n}")
        print()
        print("-" * 80)
        print()
        results = eval_list(data, samp_list, method=method, func=func, n=n)
        eval_report(results)
        print()


if __name__ == "__main__":
    main()
