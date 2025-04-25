"""
CSC 466
Dekhtyar, Fall 2023

Martin Hsu - mshsu@calpoly.edu
Brendan Callender - bscallen@calpoly.edu
"""

import pandas as pd
import numpy as np
from sys import argv

from typing import List, Dict


def eucledian_dist(d1: pd.DataFrame, d2: pd.DataFrame) -> float:
    """
    Measures the pairwise euclidean distances between two groups of points

    :param d1: A pandas dataframe representing a group of points
    :param d2: A pandas dataframe representing a group of points
    :return: A pandas series representing distances from d1 to the points in d2
    """
    return d2.apply(
        lambda row: np.sqrt(np.sum((d1.to_numpy() - row.to_numpy()) ** 2)),
        axis=1)


def update_dist(val: float, eps: float) -> float:
    """
    Turns distance into 0 if it is greater than epsilon, so it is easier to
    to check the number of epsilon neighbors

    :param val: A distance value
    :param eps: The epsilon threshold for dbscan
    :return: A float that is 0 if val > eps
    """
    if val > eps:
        return 0
    else:
        return val


def get_core_points(dist: pd.DataFrame, eps: float, minpts: int) -> bool:
    """
    Indicates if a point is a core point based on its distances from other
    points

    :param dist: A dataframe of distances
    :param eps: The epsilon threshold for dbscan
    :param minpts: Minimum points threshold for a core point for dbscan
    :return: True if the point is a core point based off its distances,
        False otherwise
    """
    if ((dist > 0).sum()) >= minpts:
        return True
    else:
        return False


def DensityConnected(dist: pd.DataFrame, pt: int, Core: List[int],
                     curr_cluster: int, clusters: List[int]) -> None:
    """
    Recursively adds all epsilon neighbors to a cluster

    :param dist: A pandas dataframe representing a distance matrix
    :param pt: Index number for point
    :param Core: List of core points
    :param curr_cluster: Number of the current cluster label
    :param clusters: A list of cluster numbers
    :return: None
    """
    N = (dist.iloc[pt,:] > 0)
    neighbors = list(N[N == True].index)
    for d in neighbors:
        if clusters[d] == -1:
            clusters[d] = curr_cluster
            if d in Core:
                DensityConnected(dist, d, Core, curr_cluster, clusters)


def DBSCAN(D: pd.DataFrame, dist: pd.DataFrame,
           eps: float, minpts: int) -> Dict:
    """
    Implements DBSCAN clustering algorithm based on Dr. Dekhtyar's pseudocode.

    :param D: Pandas dataframe where each row is a point
    :param dist: Distance matrix
    :param eps: The epsilon threshold for dbscan
    :param minpts: Minimum points threshold for a core point for dbscan
    :return: Dictionary of results including cluster groups, core points,
        noise points, border points
    """
    is_core = dist.apply(get_core_points, args=(eps, minpts), axis=0)
    core = list(is_core[is_core == True].index)

    curr_cluster = 0
    clusters = ([-1] * D.shape[0]).copy()

    for pt in core:
        if clusters[pt] == -1:
            curr_cluster += 1
            clusters[pt] = curr_cluster
            DensityConnected(dist, pt, core, curr_cluster, clusters)

    cluster_list = [[] for i in range(curr_cluster)]
    noise = []
    for j in range(len(clusters)):
        if clusters[j] != -1:
            cluster_list[clusters[j] - 1] += [j]
        else:
            noise += [j]

    d = set(list(range(D.shape[0])))
    other = set(core + noise)
    border = d.difference(other)

    return {'ClusterList': cluster_list,
            'Core': core,
            'Border': border,
            'Noise': noise}


def dist(a: pd.Series, b: pd.Series) -> float:
    """
    Finds distance between two points

    :param a: A pandas series representing a point
    :param b: A pandas series representing a point
    :return: Euclidean distance between a and b
    """
    return np.sqrt(((a-b)**2).sum())


def dbscan_report(D: pd.DataFrame, D_std: pd.DataFrame,
                  cluster_list: List[List[int]], outliers: List[int]) -> None:
    """
    Prints report for dbscan

    :param D: A pandas dataframe where each row is a point
    :param D_std: A pandas dataframe where each row is a standardized point
    :param cluster_list: List of clusters represented as list of indices
    :param outliers: List of outlier points represented by indices
    :return: None
    """
    centroids_std = D_std.groupby('Cluster').mean()
    centroids = D.groupby('Cluster').mean()
    for i, cluster in enumerate(cluster_list):
        dists = []
        a = centroids_std.loc[i]
        for pt in cluster:
            b = (D_std.iloc[pt, D_std.columns != 'Cluster'])
            dists += [dist(a, b)]
        dists = pd.Series(dists)

        print()
        print(f'Cluster: {i}')
        print(f'Center (Standardized): {", ".join([str(round(item, 5)) for item in centroids_std.loc[i].values])}')
        print(f'Center (Unstandardized): {", ".join([str(round(item, 5)) for item in centroids.loc[i].values])}')
        print(f'Max Dist. To Center (Standardized): {dists.max()}')
        print(f'Min Dist. To Center (Standardized): {dists.min()}')
        print(f'Avg Dist. To Center (Standardized): {dists.mean()}')
        print(f'Sum of Squared Error (Standardized): {(dists**2).sum()}')
        print(f'{len(cluster)} Points (Standardized):')
        for j in cluster:
            print(", ".join(list(D_std.iloc[j, D_std.columns != 'Cluster'].astype(str))))
        print(f'{len(cluster)} Points (Unstandardized):')
        for k in cluster:
            print(", ".join(list(D.iloc[k, D.columns != 'Cluster'].astype(str))))

        print()
        print("-" * 80)
    print()
    print(f'Outlier Percentage: {round(len(outliers)/len(D)*100,1)}%')
    print(f'{len(outliers)} Outliers (Standardized):')
    for l in outliers:
        print(", ".join(list(D_std.iloc[l, D_std.columns != 'Cluster'].astype(str))))
    print(f'{len(outliers)} Outliers (Unstandardized):')
    for m in outliers:
        print(", ".join(list(D.iloc[m, D.columns != 'Cluster'].astype(str))))
    print()
    print("-" * 80)


def main(argv):
    file = open(argv[1], 'r')
    cols = pd.Series(file.readline().replace('\n', '').split(','))
    file.close()

    hasLabel = False
    usecols = list(cols[cols == '1'].index)
    label = list(cols[cols == '0'].index)

    if label != []:
        hasLabel = True
        labels = pd.read_csv(argv[1], usecols=label)
    D = pd.read_csv(argv[1], usecols=usecols)

    eps = float(argv[2])
    minpts = int(argv[3])

    D_std = D.copy()
    for c in range(D.shape[1]):
        col = D.iloc[:,c]
        D_std.iloc[:,c] = (col - col.mean()) / col.std()

    dist = D_std.apply(eucledian_dist, args=(D_std,), axis=1)
    dist = dist.applymap(update_dist, eps=eps)

    C = DBSCAN(D, dist, eps, minpts)

    if hasLabel:
        D['Label'] = labels['0']

    res = [-1 for idx in range(D.shape[0])]
    for i, cluster in enumerate(C['ClusterList']):
        for elem in cluster:
            res[elem] = i
    D['Cluster'] = pd.Series(res)
    D_std['Cluster'] = pd.Series(res)

    print()
    print(f"Data: {argv[1].split('/')[-1]}")
    print(f"eps: {argv[2]}")
    print(f"minpts: {argv[3]}")
    print()
    print('Intercluster Distances (Standardized)')

    try:
        clusters = pd.DataFrame(D_std[D_std['Cluster'] != -1].groupby('Cluster').mean())
        print(clusters.apply(eucledian_dist, args=(clusters,), axis=1))
    except:
        print('No Clusters')
    print()
    print('-'*80)
    print('-'*80)

    dbscan_report(D, D_std, C['ClusterList'], C['Noise'])


if __name__ == "__main__":
    if len(argv) < 4:
        print("Syntax: python3 dbscan.py <input file> <eps> <minpts>")
    else:
        main(argv)
