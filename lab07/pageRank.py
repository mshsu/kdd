# Martin Hsu
# mshsu@calpoly.edu
# CSC 466 Fall 2023

import pandas as pd
import numpy as np
import time
import sys

from typing import Dict, List


def construct_hubs(data: pd.DataFrame,
                   directed: bool = False) -> Dict[str, List[str]]:
    """
    Turns an edgelist into an adjacency dictionary, where each key is a parent
    node and each value is a list of child nodes.

    :param data: A pandas dataframe where each row is an edge
    :param directed: (Default False) If true, treats each row in original
        dataframe as a directed edge where the first node is the child node and
        the second node is the parent node. If false, treats each row as an
        undirected edge where either node is both a parent and child node.
    :return: A dictionary, where each key is a parent node and each value is a
        list of child nodes.
    """
    if directed:
        hubs = {
            node: list(data.set_index(2).sort_index().loc[node, 0])
            if isinstance(data.set_index(2).loc[node, 0], pd.Series)
            else [data.set_index(2).loc[node, 0]]
            for node in data[2].unique()
        }
    else:
        hubs1 = {
            node: list(data.set_index(0).sort_index().loc[node, 2])
            if isinstance(data.set_index(0).loc[node, 2], pd.Series)
            else [data.set_index(0).loc[node, 2]]
            for node in data[0].unique()
        }
        hubs2 = {
            node: list(data.set_index(2).sort_index().loc[node, 0])
            if isinstance(data.set_index(2).loc[node, 0], pd.Series)
            else [data.set_index(2).loc[node, 0]]
            for node in data[2].unique()
        }
        hubs =  pd.concat([pd.Series(hubs1), pd.Series(hubs2)], axis=1)\
            .map(lambda d: d if isinstance(d, list) else []).sum(axis=1)\
            .sort_index().to_dict()

    for node in pd.concat([data[0], data[2]]).unique():
        if node not in hubs.keys():
            hubs[node] = []

    return hubs


def pagerank(hubs: Dict[str, List[str]], d: float = 0.5) -> pd.Series:
    """
    Performs PageRank algorithm.

    :param hubs: Dictionary output of construct_hubs
    :param d: A float probability between 0 and 1, inclusive
    :return: A pandas series representing PageRank scores
    """
    n = len(hubs.keys())
    matrix = pd.DataFrame(0.0, index=list(hubs.keys()),
                          columns=list(hubs.keys()))
    d_i = pd.Series(hubs).apply(len)

    for hub in hubs.keys():
        for node in hubs[hub]:
            matrix.loc[node, hub] = d / d_i.loc[hub]

    p = {0: pd.Series(hubs.keys()).apply(lambda x: 1 / n).set_axis(hubs.keys())}
    p[1] = (matrix @ p[0]) + (1 - d) / n

    i = 1
    while not np.allclose(p[i], p[i-1]):
        i += 1
        p[i] = (matrix @ p[i-1]) + (1 - d) / n

    p[i].name = i
    return p[i].sort_values(ascending=False)


def pagerank_report(ranks: pd.Series) -> None:
    """
    Prints PageRank rankings and scores

    :param ranks: Output of pagerank function
    :return: None
    """
    results = ranks.reset_index().reset_index()
    results.columns = ["Rank", "Node", "PageRank"]
    results["Rank"] = results["Rank"] + 1
    print(results.to_string(index=False))


def main():
    args = sys.argv

    directed = False
    if "--dir" in args:
        directed = True
        args.remove("--dir")

    if len(args) < 3:
        print("Syntax: python3 pageRank.py <nodeFile.csv> <d> [--dir]")
    else:
        file = args[1]
        d = float(args[2])

        start_read = time.time()
        data = pd.read_csv(file, header=None, usecols=range(4))
        data[0] = data[0].str.replace('"', '').str.strip()
        data[2] = data[2].str.replace('"', '').str.strip()
        hubs = construct_hubs(data, directed=directed)
        end_read = time.time()
        read_time = (end_read - start_read) * 1000

        start_proc = time.time()
        ranks = pagerank(hubs, d)
        end_proc = time.time()
        proc_time = (end_proc - start_proc) * 1000

        n_iter = int(ranks.name) + 1

        print(f"Page Rank, d={d:0.2f}")
        print()
        print(f"File: {file.split('/')[-1]}")
        print(f"Read Time:       {read_time:0.3f}ms")
        print(f"Processing Time: {proc_time:0.3f}ms")
        print(f"N Iterations:    {n_iter}")
        print()
        pagerank_report(ranks)


if __name__ == "__main__":
    main()
