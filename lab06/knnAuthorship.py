import pandas as pd
import numpy as np

from textVectorizer import *


def knn(vec: Vector, corp: pd.Series, k: int, metric: str = "cos") -> str:
    """
    Performs KNN on a single Vector instance.

    :param vec: A vector instance
    :param corp: A pandas series of vector instances
    :param k: Number of neighbors
    :param metric: (Default "cos") If "cos", performs KNN using cosine
        similarity. If "okapi", performs KNN using okapi
    :return: A string representing the predicted class label.
    """
    sims = None
    if metric == "cos":
        sims = corp.apply(lambda v: vec.cos_sim(v))
    elif metric == "okapi":
        sims = corp.apply(lambda v: vec.okapi(v))
    authors = corp.apply(lambda v: v.author())
    top_k_dist_ind = np.argpartition(-sims, k)[:k]
    top_k_c = authors.iloc[top_k_dist_ind]
    return top_k_c.mode().sample(frac=1).iloc[0]


def knn_predict(corp: pd.Series, k: int, metric: str = "cos") -> pd.DataFrame:
    """
    Creates KNN authorship prediction for a corpus of documents

    :param corp: A pandas series of Vector objects representing a corpus of
        documents
    :param k: Number of neighbors
    :param metric: (Default "cos") If "cos", performs KNN using cosine
        similarity. If "okapi", performs KNN using okapi
    :return: A pandas dataframe of results
    """
    n = len(corp)

    def knn_verbose(row):
        print(f"Classifying {corp.loc[row].name()}... ({row}/{n})")
        return knn(corp.loc[row], corp.drop(row), k, metric=metric)

    pred = pd.Series(corp.index).apply(
        lambda row: knn_verbose(row)
    )
    docs = corp.apply(lambda d: d.name())
    obs = corp.apply(lambda d: d.author())
    pred = pd.concat([docs, obs, pred], axis=1)
    pred.columns = ["doc", "obs", "pred"]
    return pred


def main():
    args = sys.argv
    if len(args) < 3:
        print("Syntax: python3 knnAuthorship.py <VectorFile> <k>")

    else:
        vecfile = args[1]
        homedir = vecfile.split("_")[0]
        k = int(args[2])
        print(f"Reading vector file...")
        f = pd.read_csv(vecfile, index_col=0)
        print(f"Constructing vectors...")
        corp = vectors_from_f(f)

        results = knn_predict(corp, k)
        results.to_csv(f"{homedir}_knn_results.csv", index=False)
        accuracy = (results['obs'] == results['pred']).sum() / len(results.index)

        print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
