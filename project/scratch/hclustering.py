"""
CSC 466
Dekhtyar, Fall 2023

Martin Hsu - mshsu@calpoly.edu
Brendan Callender - bscallen@calpoly.edu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

from typing import Dict, List, Optional


def link_dist(C_ij: Dict, C_ik: Dict, link: str = "average") -> float:
    """
    Calculates distance between clusters using specified method

    :param C_ij: Cluster j, with cluster in dictionary tree form
    :param C_ik: Cluster k, with cluster in dictionary tree form
    :param link: (Default "average") Type of distance between clusters to
        evaluate
    :return: Float representing distance between clusters
    """
    # Flatten tree to list of lists, where each list is a datapoint
    D_ij = pd.concat([pd.Series(leaf["data"].split(',')).astype(float)
                      for leaf in flatten(C_ij)], axis=1).T
    D_ik = pd.concat([pd.Series(leaf["data"].split(',')).astype(float)
                      for leaf in flatten(C_ik)], axis=1).T

    dists = []
    for j in D_ij.index:
        for k in D_ik.index:
            dists.append(np.sqrt(((D_ij.loc[j] - D_ik.loc[k])**2).sum()))
    dists = pd.Series(dists)

    if link == "average":
        return dists.mean()
    elif link == "single":
        return dists.min()
    elif link == "complete":
        return dists.max()
    elif link == "centroid":
        return np.sqrt(((D_ij.mean() - D_ik.mean())**2).sum())


def flatten(node: Dict) -> List[Dict]:
    """
    Flattens hcluster tree or node to a list of the tree's or node's leaves
    using recursion

    :param node: A dictionary representing the hcluster tree or node
    :return: A list of the tree/node leaves represented by a dictionary
    """
    flat = []
    if node["type"] == "leaf":
        flat.append(node)
    elif node["type"] in ("node", "root"):
        subnodes = node["nodes"]
        for n in subnodes:
            flat.extend(flatten(n))
    return flat


def make_tree(D: pd.DataFrame) -> Dict:
    """
    Constructs hcluster tree using Dr. Dekhtyar's pseudocode

    :param D: A pandas dataframe where each row is a point
    :return: A dictionary representing an hcluster tree
    """
    # Follows Dr. Dekhtyar's pseudocode below
    C = {}

    i = 1
    C[i] = [
        {"type": "leaf", "height": 0, "data": ",".join(D.loc[c].astype(str))}
        for c in list(D.index)]
    n_range = range(len(C[i]))
    d = pd.DataFrame(index=n_range, columns=n_range).astype(float)
    while len(C[i]) > 1:
        for j in n_range:
            for k in range(j + 1, len(C[i])):
                if np.isnan(d.iloc[j, k]):
                    d.iloc[j, k] = link_dist(C[i][j], C[i][k])
        d_stack = d.stack()
        s, r = d_stack.idxmin()
        height = d_stack.min()

        C[i + 1] = []
        for j in n_range:
            if j != s and j != r:
                C[i + 1].append(C[i][j])
            elif j == s:
                C[i + 1].append({"type": "node", "height": height,
                                 "nodes": [C[i][s], C[i][r]]})
                d.iloc[j] = np.nan
                d.iloc[:, j] = np.nan
        d.drop(r, axis=1, inplace=True)
        d.drop(r, inplace=True)
        n_range = range(len(C[i + 1]))
        d.columns = n_range
        d.index = n_range

        i += 1

    tree = C[i][0]
    tree["type"] = "root"
    return tree


def cut_tree(tree: Dict, threshold: float, by: str = "height") -> List[Dict]:
    """
    Cuts hcluster tree by height or number of clusters

    :param tree: A dictionary representing an hcluster tree
    :param threshold: The height or number of clusters by which to cut a tree
    :param by: (Default "height") If "height" cuts tree by height, if "nclust"
        then cuts tree by number of clusters
    :return: A list of dictionary tree nodes representing the clusters
    """
    # Uses queue: enqueue tree, dequeue first tree and enqueue its nodes
    #   until condition met
    queue = [tree]

    condition = 'False'
    if by == "height":
        condition = 'queue[0]["height"] > threshold'
    elif by == "nclust":
        condition = 'len(queue) < threshold'

    while eval(condition):
        poptree = queue.pop(0)
        queue.extend(poptree["nodes"])
        queue = sorted(queue, key=lambda n: -n["height"])
    return queue


def dist(x: pd.Series, y: pd.Series) -> float:
    """
    Computes the euclidean distance between two pandas series

    :param x: A numeric pandas series
    :param y: A numeric pandas series
    :return: A float representing the euclidean distance between x and y
    """
    return np.sqrt(((x - y)**2).sum())


def hclust(D: pd.DataFrame, threshold: float, by: str = "height",
           std: bool = True, restrict: List[int] = None,
           row_id: str = None) -> Dict:
    """
    Forms clusters by making a tree and cutting it

    :param D: A pandas dataframe where each row is a point
    :param threshold: The height or number of clusters by which to cut a tree
    :param by: (Default "height") If "height" cuts tree by height, if "nclust"
        then cuts tree by number of clusters
    :param std: (Default True) Standardizes attributes if True, does
        standardize if False
    :param restrict: (Default None) Optional list of 1s and 0s representing
        which columns to keep or drop
    :param row_id: (Default None) Name of row names column
    :return: Dictionary of kmeans results and various metrics
    """
    # Setup - create copy of data and standardize
    D_orig = D.copy()
    cols = list(D_orig.columns)
    if restrict is not None:
        cols_orig = D_orig.columns
        cols = [cols_orig[i] for i in range(len(cols_orig)) if restrict[i] == 1]
        D = D.loc[:, cols]
    D_mean, D_std = 0, 1
    if std:
        D_mean, D_std = D.mean(), D.std()
        D = (D - D_mean) / D_std
    D = D.sort_values(cols)
    D_orig = D_orig.reindex(list(D.index))

    # Make clusters by making tree and cutting it
    tree = make_tree(D)
    cut = cut_tree(tree, threshold, by=by)

    # Find predicted clusters by flattening the node clusters
    clusts = {i: [leaf["data"].split(",") for leaf in flatten(cut[i])] for i in range(len(cut))}
    clusts = {i: pd.DataFrame(clusts[i]).apply(pd.to_numeric) for i in clusts.keys()}
    for i in clusts.keys():
        clusts[i]['cluster'] = i
        clusts[i].columns = cols + ['cluster']

    # Format output:
    # Enrich standardized data with results and get index names
    D = pd.concat(clusts.values()).sort_values(cols)
    D.index = D_orig.index
    D.columns = cols + ['cluster']
    # Enrich unstandardized data with results and get index names
    D_clust = D_orig.copy()
    D_clust['cluster'] = D['cluster']
    if row_id is not None:
        D_clust = D_clust.set_index(row_id)
        D_clust.index.name = None
        D.index = D_clust.index
        D.index.name = None
    # Get centroids
    centroids = D.groupby('cluster').mean()
    # Get unstandardized centroids
    centroids_unstd = centroids * D_std + D_mean
    # Get min, max, avg error and sse
    err = D.apply(lambda x: dist(x.drop('cluster'), centroids.loc[x.loc['cluster']]), axis=1)
    err = pd.DataFrame({'err': err, 'cluster': D['cluster']})
    err['err_sq'] = err['err']**2
    err = err.groupby('cluster')
    err_stats = pd.DataFrame({'min': err['err'].min(),
                              'max': err['err'].max(),
                              'avg': err['err'].mean(),
                              'sse': err['err_sq'].sum()})
    # Get intercluster distances
    interclust = pd.DataFrame(index=list(centroids.index), columns=list(centroids.index))
    for i in interclust.index:
        for j in interclust.columns:
            interclust.loc[i, j] = dist(centroids.loc[i], centroids.loc[j])

    return {'clustered original data': D_clust.sort_index(),
            'clustered processed data': D.sort_index(),
            'tree': tree,
            'threshold': threshold,
            'cut type': by,
            'standardized': std,
            'centroids': centroids,
            'unstandardized centroids': centroids_unstd,
            'distance summary': err_stats,
            'intercluster distances': interclust}


def hclust_report(results, header: bool = False,
                  row_id: bool = False) -> None:
    """
    Prints hclustering report

    :param results: Dictionary output from hclust function
    :param header: (Default False) True if the data has column names, False
        if not
    :param row_id: (Default False) True if there are row names, False if not
    :return: None
    """
    orig = results['clustered original data']
    proc = results['clustered processed data']
    threshold = results["threshold"]
    by = results["cut type"]
    std = results['standardized']
    c = results['centroids']
    c_unstd = results['unstandardized centroids']
    err_stats = results['distance summary']
    interclust = results['intercluster distances']
    std_str = ""
    if std:
        std_str = " (Standardized)"
    k = len(c.index)

    print(f"Number of Clusters: {k}")
    if by == "height":
        print(f"Cut Height: {threshold}")
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


def hclust_plot(results: Dict, cols: Optional[List[int]] = None) -> None:
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
    plt.title("HCLUST")


def main():
    args = sys.argv

    std = True
    if "--nostd" in args:
        std = False
        args.remove("--nostd")
    output = True
    if "--notree" in args:
        output = False
        args.remove("--notree")
    by = "height"
    if "--nclust" in args:
        by = "nclust"
        args.remove("--nclust")

    if len(args) < 2:
        print("Syntax: python3 hclustering.py <Filename.csv> <threshold> "
              "[--id <rowId>] [--header <headerFile.txt>] [--nclust] [--nostd] "
              "[--notree]")

    else:
        filename = args[1]
        threshold = float(args[2])
        printout = True
        if threshold == -1:
            printout = False
            threshold = 0

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

        cluster_results = hclust(data, threshold, by=by,
                                 std=std, restrict=restrict, row_id=row_id)
        json_data = json.dumps(cluster_results['tree'], indent=1)
        if printout:
            print(f"Data: {filename.split('/')[-1]}")
            hclust_report(cluster_results,
                       header=(header is not None), row_id=(row_id is not None))

        if output:
            outpath = filename.split('/')[-1].replace(".csv", "") + "_tree.json"
            with open(outpath, 'w') as json_file:
                json_file.write(json_data)


if __name__ == "__main__":
    main()
