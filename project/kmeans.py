"""
CSC 466
Dekhtyar, Fall 2023

Martin Hsu - mshsu@calpoly.edu
Brendan Callender - bscallen@calpoly.edu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from typing import List, Dict, Optional


def dist(x: pd.Series, m: Dict[int, List]) -> pd.Series:
    """
    Computes euclidean distances between a point and centroids.

    :param x: A pandas series representing a point
    :param m: A dictionary of list where each list represents a centroid
    :return: A pandas series of distances between the point and centroids
    """
    dists = {i: np.sqrt(((x.reset_index(drop=True) - pd.Series(m[i]))**2).sum())
             for i in m.keys()}
    return pd.Series(dists)


def select_initial_centroids(D: pd.DataFrame, k: int) -> Dict[int, List]:
    """
    Selects initial centroids by selecting furthest separated points with an
    algorithm.

    :param D: A pandas dataframe where each row is a point
    :param k: Number of means
    :return: Dictionary of lists representing initial centroids
    """
    centroid_list = {0: D.mean().tolist()}
    for i in range(k):
        c = D.apply(lambda x: dist(x, centroid_list).sum(), axis=1).idxmax()
        centroid_list[i] = D.loc[c].tolist()
        D = D.drop(c, axis=0)

    return centroid_list


def is_stopping_condition(m: Dict[int, List], m_new: Dict[int, List],
                          cl: Dict[int, List], cl_new: Dict[int, List]) -> bool:
    """
    Checks for kmeans algorithm stopping conditions

    :param m: Dictionary of lists representing old centroids
    :param m_new: Dictionary of lists representing new centroids
    :param cl: Dictionary of lists representing the point indices in each
        old cluster
    :param cl_new: Dictionary of lists representing the point indices in each
        new cluster
    :return:
    """
    # If no old cluster, don't stop
    if cl is None:
        return False

    # First stopping condition: If cluster points stop changing, stop
    stop1 = all([set(cl[key]) == set(cl_new[key]) for key in cl.keys()])
    # 2nd stopping condition: If centroids change minimally, stop
    stop2 = all([np.allclose(m[key], m_new[key]) for key in m.keys()])

    return stop1 or stop2


def kmeans(D: pd.DataFrame, k: int, std: bool = True,
           restrict: Optional[List[int]] = None,
           row_id: Optional[str] = None) -> Dict:
    """
    Performs kmeans algorithm on data according to Dr. Dekhtyar's pseudocode.

    :param D: A pandas dataframe where each row is a point
    :param k: Number of means
    :param std: (Default True) Standardizes attributes if True, does
        standardize if False
    :param restrict: (Default None) Optional list of 1s and 0s representing
        which columns to keep or drop
    :param row_id:  (Default None) Name of row names column
    :return: Dictionary of kmeans results and various metrics
    """
    # Setup - create copy of data and standardize
    D_orig = D.copy()
    if restrict is not None:
        cols_orig = D_orig.columns
        cols = [cols_orig[i] for i in range(len(cols_orig)) if restrict[i] == 1]
        D = D.loc[:, cols]
    D_mean, D_std = 0, 1
    if std:
        D_mean, D_std = D.mean(), D.std()
        D = (D - D_mean) / D_std

    # Dr. Dekhtyar's pseudocode here
    m = select_initial_centroids(D, k)
    s, num, cl = {}, {}, None
    err = {}
    cl_new, m_new = {}, {}

    stop = False
    while not stop:
        for j in range(k):
            s[j] = pd.Series([0] * len(D.columns))
            s[j].index = D.columns
            num[j] = 0
            cl_new[j] = []
        for i in D.index:
            x = D.loc[i]
            dists = dist(x, m)
            cluster = dists.idxmin()
            err[i] = dists.min()
            cl_new[cluster].append(i)
            s[cluster] = s[cluster] + x
            num[cluster] += 1
        for j in range(k):
            m_new[j] = (s[j] / max(num[j], 1)).tolist()
        stop = is_stopping_condition(m, m_new, cl, cl_new)
        cl, m = cl_new.copy(), m_new.copy()

    # Format output:
    # Get final cluster predictions in proper order
    final_cl = []
    for key in cl.keys():
        cl_i = pd.Series([key] * len(cl[key]))
        cl_i.index = cl[key]
        final_cl.append(cl_i)
    final_cl = pd.concat(final_cl).sort_index()
    # Get centroids
    centroids = pd.DataFrame.from_dict(m, orient='index')
    centroids.columns = D.columns
    # Get unstandardized centroids
    centroids_unstd = centroids * D_std + D_mean
    # Enrich standardized data with results and get index names
    D['cluster'] = final_cl
    if row_id is not None:
        D = D.set_index(D_orig[row_id])
        D.index.name = None
    # Enrich unstandardized data with results and get index names
    D_clust = D_orig.copy()
    D_clust['cluster'] = final_cl
    if row_id is not None:
        D_clust = D_clust.set_index(row_id)
        D_clust.index.name = None
    # Get min, max, avg error and sse
    err = pd.DataFrame({'err': pd.Series(err).sort_index(),
                        'cluster': final_cl})
    err['err_sq'] = err['err']**2
    err = err.groupby('cluster')
    err_stats = pd.DataFrame({'min': err['err'].min(),
                              'max': err['err'].max(),
                              'avg': err['err'].mean(),
                              'sse': err['err_sq'].sum()})
    # Get intercluster distances
    interclust = pd.DataFrame(
        {i: dist(centroids.loc[i], m) for i in centroids.index}
    )

    return {'clustered original data': D_clust,
            'clustered processed data': D,
            'k': k,
            'standardized': std,
            'centroids': centroids,
            'unstandardized centroids': centroids_unstd,
            'distance summary': err_stats,
            'intercluster distances': interclust}


def kmeans_report(results: Dict, header: bool = False,
                  row_id: bool = False) -> None:
    """
    Prints kmeans report

    :param results: Dictionary output from kmeans function
    :param header: (Default False) True if the data has column names, False
        if not
    :param row_id: (Default False) True if there are row names, False if not
    :return: None
    """
    orig = results['clustered original data']
    proc = results['clustered processed data']
    k = results['k']
    std = results['standardized']
    c = results['centroids']
    c_unstd = results['unstandardized centroids']
    err_stats = results['distance summary']
    interclust = results['intercluster distances']
    std_str = ""
    if std:
        std_str = " (Standardized)"

    print(f"Number of Clusters: {k}")
    print(f"Standardized Data: {std}")
    print()
    print(f"Intercluster Distances{std_str}:")
    print(interclust.to_string())
    print()
    print("-" * 80)
    for i in range(k):
        print("-" * 80)
        print()
        print(f"Cluster: {i}")
        print(f"Cluster Size: {len(orig[orig['cluster'] == i].index)}")
        if header and std:
            print(f"Center (Standardized):")
            print(c.loc[i].to_frame().T.to_string(index=False))
            print(f"Center (Unstandardized)")
            print(c_unstd.loc[i].to_frame().T.to_string(index=False))
        elif not header and std:
            print(f"Center (Standardized): "
                  f"{', '.join(list(c.loc[i].astype(str)))}")
            print(f"Center (Unstandardized): "
                  f"{', '.join(list(c_unstd.loc[i].astype(str)))}")
        elif header and not std:
            print(f"Center:")
            print(c.loc[i].to_frame().T.to_string(index=False))
        elif not header and not std:
            print(f"Center: "
                  f"{', '.join(list(c.loc[i].astype(str)))}")
        print(f"Max Dist. to Center{std_str}:  {err_stats.loc[i, 'max']}")
        print(f"Min Dist. to Center{std_str}:  {err_stats.loc[i, 'min']}")
        print(f"Avg Dist. to Center{std_str}:  {err_stats.loc[i, 'avg']}")
        print(f"Sum of Squared Error{std_str}: {err_stats.loc[i, 'sse']}")
        # n_clust = len(orig[orig['cluster'] == i].index)
        # if std:
        #     print(f"{n_clust} Points (Standardized):")
        #     proc_report = proc[proc['cluster'] == i] \
        #         .drop('cluster', axis=1)
        #     if header and row_id:
        #         print(proc_report.to_string())
        #     elif header:
        #         print(proc_report.to_string(index=False))
        #     elif row_id:
        #         print(proc_report.to_string(header=False))
        #     else:
        #         for j in range(n_clust):
        #             print(", ".join(list(proc_report.iloc[j].astype(str))))
        #     print(f"{n_clust} Points (Unstandardized):")
        #     orig_report = orig[orig['cluster'] == i]\
        #         .drop('cluster', axis=1)
        #     if header and row_id:
        #         print(orig_report.to_string())
        #     elif header:
        #         print(orig_report.to_string(index=False))
        #     elif row_id:
        #         print(orig_report.to_string(header=False))
        #     else:
        #         for j in range(n_clust):
        #             print(", ".join(list(orig_report.iloc[j].astype(str))))
        # else:
        #     print(f"{n_clust} Points:")
        #     orig_report = orig[orig['cluster'] == i]\
        #         .drop('cluster', axis=1)
        #     if header and row_id:
        #         print(orig_report.to_string())
        #     elif header:
        #         print(orig_report.to_string(index=False))
        #     elif row_id:
        #         print(orig_report.to_string(header=False))
        #     else:
        #         for j in range(n_clust):
        #             print(", ".join(list(orig_report.iloc[j].astype(str))))
        print()


def kmeans_plot(results: Dict, cols: Optional[List[int]] = None) -> None:
    df = results['clustered original data']
    if cols is not None:
        df = df.iloc[:, cols + [-1]]
    dim = len(df.columns) - 1
    if dim == 2:
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df.iloc[:, -1])
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2],
                     c=df.iloc[:, -1])
    plt.title("KMEANS")


def main():
    args = sys.argv

    std = True
    if "--nostd" in args:
        std = False
        args.remove("--nostd")

    if len(args) < 3:
        print("Syntax: python3 kmeans.py <Filename.csv> <k> "
              "[--id <rowId>] [--header <headerFile.txt>] [--nostd]")
    else:
        filename = args[1]
        k = int(args[2])

        header = None
        row_id = None
        if len(args) in (5, 7):
            option = args[3]
            if option == "--id":
                try:
                    row_id = int(args[4])
                except ValueError:
                    row_id = args[4]
            elif option == "--header":
                hfilepath = args[4]
                with open(hfilepath, 'r') as hfile:
                    header = hfile.read().replace("\n", "").split(",")
        if len(args) == 7:
            option = args[5]
            if option == "--id":
                try:
                    row_id = int(args[6])
                except ValueError:
                    row_id = args[6]
            elif option == "--header":
                hfilepath = args[6]
                with open(hfilepath, 'r') as hfile:
                    header = hfile.read().replace("\n", "").split(",")

        restrict = pd.read_csv(filename, header=None).loc[0].tolist()
        data = pd.read_csv(filename, header=None, skiprows=1)
        if header is not None:
            data.columns = header

        cluster_results = kmeans(data, k,
                                 std=std, restrict=restrict, row_id=row_id)
        print(f"Data: {filename.split('/')[-1]}")
        kmeans_report(cluster_results,
                      header=(header is not None), row_id=(row_id is not None))


if __name__ == "__main__":
    main()
