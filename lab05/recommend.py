# CSC 466 Fall 2023
# Martin Hsu - mshsu@calpoly.edu
# Rachel Roggenkemper - rroggenk@calpoly.edu
"""
Base code with all the collaborative filtering algorithms
"""

import pandas as pd
import numpy as np

from typing import Optional, Callable


def pearson_corr(x1: pd.Series, x2: pd.Series) -> float:
    """
    Calculates Pearson Correlation Coefficient between two pandas series'.

    :param x1: A pandas series
    :param x2: A pandas series
    :return: A float representing the Pearson Correlation Coefficient between
        x1 and x2
    """
    # Calculate centered values
    x1_cent = x1 - x1.mean()
    x2_cent = x2 - x2.mean()

    # Calculate numerator
    num = (x1_cent * x2_cent).sum()
    # Calculate denominator
    denom = np.sqrt((x1_cent ** 2).sum()) * np.sqrt((x2_cent ** 2).sum())

    return num / denom


def similarity_matrix(data: pd.DataFrame,
                      method: str = "user") -> pd.DataFrame:
    """
    Creates a Pearson Correlation Coefficient matrix between either users or
    items in a ratings datatable

    :param data: A pandas dataframe representing ratings data from which to
        create the matrix
    :param method: (Default "user") A string, either "user" or "item," which
        denotes which dimension upon which to create the matrix
    :return: A pandas dataframe representing a Pearson Correlation Coefficient
        matrix
    """
    # Initialize similarity matrix
    sim = pd.DataFrame()
    # User-based method
    if method == "user":
        # Similarity matrix is square matrix with users in rows/cols
        users = list(data.index)
        sim = pd.DataFrame(index=data.index, columns=data.index)
        # Populate similarity matrix with Pearson coefs
        for i in users:
            for j in users:
                if i == j:
                    # We want user of interest to be excluded from calculations
                    #   in the CF algorithms. We accomplish this by putting NaN.
                    sim.loc[i, j] = np.nan
                else:
                    sim.loc[i, j] = pearson_corr(data.loc[i, :], data.loc[j, :])
    # Item-based method
    elif method == "item":
        # Similarity matrix is square matrix with items in rows/cols
        items = list(data.columns)
        sim = pd.DataFrame(index=data.columns, columns=data.columns)
        # Populate similarity matrix with Pearson coefs
        for i in items:
            for j in items:
                if i == j:
                    # We want item of interest to be excluded from calculations
                    #   in the CF algorithms. We accomplish this by putting NaN.
                    sim.loc[i, j] = np.nan
                else:
                    sim.loc[i, j] = pearson_corr(data.loc[:, i], data.loc[:, j])

    return sim


def mean_utility(orig: pd.DataFrame, sim: pd.DataFrame,
                 user: int, item: int,
                 method: str = "user", n: Optional[int] = None) -> float:
    """
    Predicts rating for a given user and item in a ratings database using the
        mean utility method.

    :param orig: The original ratings database. Missing values must be denoted
        by NaN.
    :param sim: The similarity matrix for orig
    :param user: The user for which to predict rating
    :param item: The item for which to predict rating
    :param method: (Default "user") A string, either "user" or "item," which
        specifies whether to use user-based or item-based recommendations.
    :param n: (Default None) An optional integer, where if specified will denote
        the number of neighbors with which to calculate the rating.
    :return: A predicted rating for a given user and item in the orig ratings
        database.
    """
    # User-based method
    if method == "user":
        # Get all users for specified item
        users_of_item = orig.loc[:, item].copy()
        # We want to exclude user of interest from calculations regardless if
        #   it is missing or not, so either way we set it to NaN
        users_of_item.loc[user] = np.nan
        # Use N-Nearest-Neighbors if n is not None
        if n is not None:
            # For user of interest, get similarities with other users that have
            #   a non-missing rating for the item of interest
            users_sim = sim.loc[user][~users_of_item.isna()].copy()
            # Partially sort and get N-nearest-neighbors based on similarity
            neighbors = pd.Series(users_sim.index)[
                np.argpartition(-users_sim, n)[:n]]
            # Now, we aggregate on only the nearest neighbors
            users_of_item = users_of_item.loc[neighbors]
        # Return result, but keep between 10 and -10
        return max(min(users_of_item.mean(), 10), -10)

    # Item-based method
    elif method == "item":
        # Get all items for specified user
        items_of_user = orig.loc[user, :].copy()
        # We want to exclude item of interest from calculations regardless if
        #   it is missing or not, so either way we set it to NaN
        items_of_user.loc[item] = np.nan
        # Use N-Nearest-Neighbors if n is not None
        if n is not None:
            # For item of interest, get similarities with other items that have
            #   a non-missing rating for the user of interest
            items_sim = sim.loc[item][~items_of_user.isna()].copy()
            # Partially sort and get N-nearest-neighbors based on similarity
            neighbors = pd.Series(items_sim.index)[
                np.argpartition(-items_sim, n)[:n]]
            # Now, we aggregate on only the nearest neighbors
            items_of_user = items_of_user.loc[neighbors]
        # Return result, but keep between 10 and -10
        return max(min(items_of_user.mean(), 10), -10)

    return np.nan


def wtd_sum(orig: pd.DataFrame, sim: pd.DataFrame,
            user: int, item: int,
            method: str = "user", n: Optional[int] = None) -> float:
    """
    Predicts rating for a given user and item in a ratings database using the
        weighted sum method.

    :param orig: The original ratings database. Missing values must be denoted
        by NaN.
    :param sim: The similarity matrix for orig
    :param user: The user for which to predict rating
    :param item: The item for which to predict rating
    :param method: (Default "user") A string, either "user" or "item," which
        specifies whether to use user-based or item-based recommendations.
    :param n: (Default None) An optional integer, where if specified will denote
        the number of neighbors with which to calculate the rating.
    :return: A predicted rating for a given user and item in the orig ratings
        database.
    """
    # User-based method
    if method == "user":
        # Get all users for specified item
        users_of_item = orig.loc[:, item].copy()
        # We want to exclude user of interest from calculations regardless if
        #   it is missing or not, so either way we set it to NaN
        users_of_item.loc[user] = np.nan
        # Get similarities to use in final calculation
        users_sim = sim.loc[user].copy()
        # Use N-Nearest-Neighbors if n is not None
        if n is not None:
            # For user of interest, get similarities with other users that have
            #   a non-missing rating for the item of interest
            users_sim = users_sim[~users_of_item.isna()].copy()
            # Partially sort and get N-nearest-neighbors based on similarity
            neighbors = pd.Series(users_sim.index)[
                np.argpartition(-users_sim, n)[:n]]
            # Now, we aggregate on only the nearest neighbors
            users_of_item = users_of_item.loc[neighbors]
            users_sim = users_sim[neighbors]
        # Return result, but keep between 10 and -10
        return max(min((users_of_item * users_sim).sum() / users_sim.sum(), 10),
                   -10)

    # Item-based method
    elif method == "item":
        # Get all items for specified user
        items_of_user = orig.loc[user, :].copy()
        # We want to exclude item of interest from calculations regardless if
        #   it is missing or not, so either way we set it to NaN
        items_of_user.loc[item] = np.nan
        # Get similarities to use in final calculation
        items_sim = sim.loc[item].copy()
        # Use N-Nearest-Neighbors if n is not None
        if n is not None:
            # For item of interest, get similarities with other items that have
            #   a non-missing rating for the user of interest
            items_sim = items_sim[~items_of_user.isna()].copy()
            # Partially sort and get N-nearest-neighbors based on similarity
            neighbors = pd.Series(items_sim.index)[
                np.argpartition(-items_sim, n)[:n]]
            # Now, we aggregate on only the nearest neighbors
            items_of_user = items_of_user.loc[neighbors]
            items_sim = items_sim[neighbors]
        # Return result, but keep between 10 and -10
        return max(min((items_of_user * items_sim).sum() / items_sim.sum(), 10),
                   -10)

    return np.nan


def adj_wtd_sum(orig: pd.DataFrame, sim: pd.DataFrame,
                user: int, item: int,
                method: str = "user", n: Optional[int] = None) -> float:
    """
    Predicts rating for a given user and item in a ratings database using the
        adjusted weighted sum method.

    :param orig: The original ratings database. Missing values must be denoted
        by NaN.
    :param sim: The similarity matrix for orig
    :param user: The user for which to predict rating
    :param item: The item for which to predict rating
    :param method: (Default "user") A string, either "user" or "item," which
        specifies whether to use user-based or item-based recommendations.
    :param n: (Default None) An optional integer, where if specified will denote
        the number of neighbors with which to calculate the rating.
    :return: A predicted rating for a given user and item in the orig ratings
        database.
    """
    # User-based method
    if method == "user":
        # Get all users for specified item
        users_of_item = orig.loc[:, item].copy()
        # We want to exclude user of interest from calculations regardless if
        #   it is missing or not, so either way we set it to NaN
        users_of_item.loc[user] = np.nan
        # Get means to use in calculation
        user_means = orig.mean(axis=1)
        user_mean = user_means.loc[user]
        # Get similarities to use in final calculation
        users_sim = sim.loc[user].copy()
        # Use N-Nearest-Neighbors if n is not None
        if n is not None:
            # For user of interest, get similarities with other users that have
            #   a non-missing rating for the item of interest
            users_sim = users_sim[~users_of_item.isna()].copy()
            # Partially sort and get N-nearest-neighbors based on similarity
            neighbors = pd.Series(users_sim.index)[
                np.argpartition(-users_sim, n)[:n]]
            # Now, we aggregate on only the nearest neighbors
            users_of_item = users_of_item.loc[neighbors]
            user_means = user_means.loc[neighbors]
            users_sim = users_sim[neighbors]
        # Return result, but keep between 10 and -10
        return max(min(user_mean + (((users_of_item - user_means) * users_sim)
                                    .sum() / users_sim.sum()), 10), -10)

    # Item-based method
    elif method == "item":
        # Get all items for specified user
        items_of_user = orig.loc[user, :].copy()
        # We want to exclude item of interest from calculations regardless if
        #   it is missing or not, so either way we set it to NaN
        items_of_user.loc[item] = np.nan
        # Get means to use in calculation
        item_means = orig.mean(axis=0)
        item_mean = item_means.loc[item]
        # Get similarities to use in final calculation
        items_sim = sim.loc[item].copy()
        # Use N-Nearest-Neighbors if n is not None
        if n is not None:
            # For item of interest, get similarities with other items that have
            #   a non-missing rating for the user of interest
            items_sim = items_sim[~items_of_user.isna()].copy()
            # Partially sort and get N-nearest-neighbors based on similarity
            neighbors = pd.Series(items_sim.index)[
                np.argpartition(-items_sim, n)[:n]]
            # Now, we aggregate on only the nearest neighbors
            items_of_user = items_of_user.loc[neighbors]
            item_means = item_means.loc[neighbors]
            items_sim = items_sim[neighbors]
        # Return result, but keep between 10 and -10
        return max(min(item_mean + (((items_of_user - item_means) * items_sim)
                                    .sum() / items_sim.sum()), 10), -10)

    return np.nan


def recommend_all(data: pd.DataFrame, method: str = "user",
                  func: Callable = mean_utility,
                  n: Optional[int] = None) -> pd.DataFrame:
    """
    Performs predicts ratings for entire ratings database. This function is not
    used in reporting and is for exploratory and debugging purposes only.

    :param data: A pandas dataframe representing the original ratings database
    :param method: (Default "user") A string, either "user" or "item," which
        specifies whether to use user-based or item-based recommendations.
    :param func: (Default mean_utility) Function with which to perform ratings
        predictions
    :param n: (Default None) An optional integer, where if specified will denote
        the number of neighbors with which to calculate the rating.
    :return: A pandas dataframe of predicted ratings for entire ratings database
    """
    # Initialize prediction results dataframe
    results = pd.DataFrame(index=data.index, columns=data.columns)
    # Replace 99's with NaNs
    orig = data.replace(99, np.nan)

    # Calculate similarity matrix
    sim = similarity_matrix(data, method=method)
    # Populate predictions
    for user in orig.index:
        for item in orig.columns:
            results.loc[user, item] = func(orig, sim, user, item, method, n=n)

    return results


def decision(rating: float) -> bool:
    """
    Given a rating, returns a recommendation. To be vectorized over a pandas
    dataframe.

    :param rating: A float representing a rating.
    :return: True if ratings >= 5, False otherwise.
    """
    return rating >= 5
