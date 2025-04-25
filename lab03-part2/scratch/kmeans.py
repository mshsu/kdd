import pandas as pd
import numpy as np

from typing import List, Dict, Optional


def dist(x: pd.Series, m: Dict[int, List]) -> pd.Series:
    x_num = x.tolist()
    dists = {i: np.sqrt(((pd.Series(x_num) - pd.Series(m[i]))**2).sum())
             for i in m.keys()}
    return pd.Series(dists)


def select_initial_centroids(D: pd.DataFrame, k: int) -> Dict[int, List]:
    centroid_list = {0: D.mean().tolist()}
    for i in range(k):
        c = D.apply(lambda x: dist(x, centroid_list).sum(), axis=1).idxmax()
        centroid_list[i] = D.loc[c].tolist()
        D = D.drop(c, axis=0)

    return centroid_list


def dummify(df: pd.DataFrame) -> pd.DataFrame:
    df_dummy = df.copy()
    cat_cols = [col for col in df.columns
                if pd.api.types.is_object_dtype(df[col])]

    for col in cat_cols:
        cats = set(df[col])
        for cat in cats:
            df_dummy[col + '_' + str(cat)] = (df[col] == cat) * 1
        df_dummy.drop(col, axis=1, inplace=True)

    return df_dummy


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df_std = df.copy()
    num_cols = [col for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col])]

    for col in num_cols:
        df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()

    return df_std


def is_stopping_condition(m: Dict[int, List], m_new: Dict[int, List],
                          cl: Dict[int, List], cl_new: Dict[int, List]) -> bool:
    if cl is None:
        return False

    stop1 = all([set(cl[key]) == set(cl_new[key]) for key in cl.keys()])
    stop2 = all([np.allclose(m[key], m_new[key]) for key in m.keys()])

    return stop1 or stop2


def knn(D: pd.DataFrame, k: int, std: bool = False,
        classvar: Optional[str] = None):
    D_orig = D.copy()
    if classvar is not None:
        D = D.drop(classvar, axis=1)
    if std:
        D = standardize(D)
    D = dummify(D)
    m = select_initial_centroids(D, k)
    s, num, cl = {}, {}, None
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
            cl_new[cluster].append(i)
            s[cluster] = s[cluster] + x
            num[cluster] += 1
        for j in range(k):
            m_new[j] = (s[j] / num[j]).tolist()
        stop = is_stopping_condition(m, m_new, cl, cl_new)
        cl, m = cl_new.copy(), m_new.copy()

    final_cl = []
    for key in cl.keys():
        cl_i = pd.Series([key] * len(cl[key]))
        cl_i.index = cl[key]
        final_cl.append(cl_i)
    final_cl = pd.concat(final_cl).sort_index()
    D_orig['cluster'] = final_cl
    return D_orig


def main():
    pass


if __name__ == "__main__":
    main()
