import pandas as pd
import numpy as np
import json

from textVectorizer import *
from typing import Tuple, Optional, List, Dict


def entropy(D: pd.DataFrame, C: pd.Series,
            A_i: Optional[str] = None,
            x: Optional[float] = None) -> float:
    """
    Calculates entropy of a dataset or attribute.

    :param D: A pandas dataframe of the predictor attributes and their values.
    :param C: A pandas series of values of the class attribute.
    :param A_i: (Default None) String name of attribute to calculate entropy
        for. If specified, function will calculate weighted average entropy over
        the attribute labels. If not, will calculate simple entropy for the
        entire dataset D.
    :param x: (Default None) Splitting value for numeric attributes. Must
        include if A_i is the name of a numeric attribute.
    :return: A float representing the entropy of the dataset or attribute.
    """
    e = 0
    # Simple entropy
    if A_i is None and x is None:
        e = C.value_counts(normalize=True).apply(
            lambda p: -p * np.log2(p)).sum()
    # Weighted avg entropy over a categorical attribute
    elif x is None:
        prop = D[A_i].value_counts(normalize=True)
        for a in list(prop.index):
           e += prop.loc[a] * entropy(D[D[A_i] == a], C[D[A_i] == a])
    # Weighted avg entropy over a numeric attribute split by value x
    else:
        D_minus, C_minus = D[D[A_i] <= x], C[D[A_i] <= x]
        D_plus, C_plus = D[D[A_i] > x], C[D[A_i] > x]

        prop_minus = len(D_minus.index) / len(D.index)
        prop_plus = len(D_plus.index) / len(D.index)

        e = (prop_minus * entropy(D_minus, C_minus) +
             prop_plus * entropy(D_plus, C_plus))
    return e


def findBestSplit(A_i: str, D: pd.DataFrame, C: pd.Series) -> float:
    """
    Finds best splitting value of a numeric attribute

    :param A_i: Name of numeric attribute
    :param D: A pandas dataframe of the predictor attributes and their values.
    :param C: A pandas series of values of the class attribute.
    :return: Best splitting value float
    """
    # Didn't follow the pseudocode because pandas can do it in fewer steps
    p0 = entropy(D, C)
    calc = pd.DataFrame()
    calc['alpha'] = D[A_i].unique()
    calc['entropy'] = calc['alpha'].apply(lambda x: entropy(D, C, A_i, x))
    calc['gain'] = p0 - calc['entropy']
    calc = calc.set_index('alpha')

    return calc['gain'].idxmax()


def find_most_frequent_label(C: pd.Series) -> Tuple[str, float]:
    """
    Finds most frequent label from the values of an attribute and returns the
    value of label and its relative frequency.

    :param C: A pandas series of values of an attribute.
    :return: A tuple where the first entry is the most frequent label, and the
        second entry is the label's relative frequency.
    """
    prop = C.value_counts(normalize=True)
    return str(prop.idxmax()), prop.max()


def selectSplittingAttribute(A: List[str], D: pd.DataFrame, C: pd.Series,
                             threshold: float,
                             gratio: bool = False) \
        -> Optional[Tuple[str, Optional[float]]]:
    """
    Selects ideal splitting attribute given a list of attributes, a dataframe
    of the attributes and their values, the values of the class attribute, and
    a threshold.

    :param A: A list of attribute names.
    :param D: A pandas dataframe of the predictor attributes and their values.
    :param C: A pandas series of values of the class attribute.
    :param threshold: A float representing a limiting threshold for the info
        gain.
    :param gratio: (Default False) If True, uses the info gain ratio instead of
        the info gain to evaluate an ideal splitting attribute.
    :return: The name of the ideal splitting attribute.
    """
    # Follows Dr. Dekhtyar's pseudocode
    p = {}
    gain = {}
    x = {}
    p[0] = entropy(D, C)
    for A_i in A:
        if pd.api.types.is_numeric_dtype(D[A_i]):
            x[A_i] = findBestSplit(A_i, D, C)
            p[A_i] = entropy(D, C, A_i, x[A_i])
        else:
            p[A_i] = entropy(D, C, A_i)
        gain[A_i] = p[0] - p[A_i]
        if gratio:
            denom = D[A_i].value_counts(normalize=True).apply(
                lambda pr: -pr * np.log2(pr)).sum()
            # Included to handle zero division cases
            if gain[A_i] != 0 and denom != 0:
                gain[A_i] = gain[A_i] / denom
            elif gain[A_i] == 0:
                gain[A_i] = 0
            elif denom == 0:
                gain[A_i] = np.infty
    best = max(gain, key=gain.get)
    if gain[best] > threshold:
        if best in x.keys():
            return best, x[best]
        else:
            return best, None
    else:
        return None


def C45(D: pd.DataFrame, A: dict, C: pd.Series,
        threshold: float, gratio: bool = False) -> dict:
    """
    Implements the C45 algorithm to construct a decision tree classifier.

    :param D: A pandas dataframe of the predictor attributes and their values.
    :param A: A dictionary where each key is the name of a predictor attribute
        and the value is the list of the attribute's unique values.
    :param C: A pandas series of values of the class attribute.
    :param threshold: A float representing a limiting threshold for the info
        gain.
    :param gratio: (Default False) If True, uses the info gain ratio instead of
        the info gain to evaluate an ideal splitting attribute.
    :return: A dictionary representing a decision tree fit to the training data.
    """
    # Follows Dr. Dekhtyar's pseudocode
    T = {"dataset": ""}
    if len(C.unique()) == 1:
        T["leaf"] = {"decision": C.unique()[0], 'p': 1}
    elif len(A) == 0:
        c, p = find_most_frequent_label(C)
        T["leaf"] = {"decision": c, "p": p}
    else:
        Ag_alpha = selectSplittingAttribute(list(A.keys()), D, C,
                                            threshold, gratio=gratio)
        if Ag_alpha is None:
            c, p = find_most_frequent_label(C)
            T["leaf"] = {"decision": c, "p": p}
        else:
            A_g, alpha = Ag_alpha
            r = {"var": A_g, "edges": []}
            T["node"] = r
            if alpha is not None:
                V = ["<=" + str(alpha), ">" + str(alpha)]
            else:
                V = A[A_g]
            for v in V:
                if alpha is None:
                    D_v = D[D[A_g] == v]
                    C_v = C[D[A_g] == v]
                else:
                    # Resolves to D[D[A_g] <= alpha] or D[D[A_g] > alpha]
                    D_v = D[eval("D[A_g]" + v)]
                    C_v = C[eval("D[A_g]" + v)]
                if len(D_v.index) != 0:
                    A_v = A.copy()
                    if pd.api.types.is_object_dtype(D[A_g]):
                        del A_v[A_g]
                    T_v = C45(D_v, A_v, C_v, threshold)
                    new_edge = {"value": v}
                    if "node" in T_v.keys():
                        new_edge["node"] = T_v["node"]
                    elif "leaf" in T_v.keys():
                        new_edge["leaf"] = T_v["leaf"]
                    r["edges"].append(
                        {"edge": new_edge}
                    )
                else:
                    c, p = find_most_frequent_label(C)
                    r["edges"].append(
                        {"edge": {"value": v,
                                  "leaf": {"decision": c, "p": p}}}
                    )
    return T


def search_tree(row: pd.Series, tree: dict) -> Optional[str]:
    """
    Recursively searches our tree until we hit a leaf.

    :param row: row of dataframe
    :param tree: decision tree
    :return: decision generated from our tree
    """
    subtree = tree
    while "leaf" not in subtree.keys():
        node = subtree["node"]
        label = row[node['var']]
        for edge in node["edges"]:
            value = edge['edge']['value']
            if np.isreal(label) and eval(str(label) + value):
                subtree = edge['edge']
            elif value == label:
                subtree = edge['edge']
    return subtree['leaf']['decision']


def dataset_selection(D: pd.DataFrame, A: dict, C: pd.Series,
                      num_attrs: int,
                      num_obs: int) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Bootstrap samples from data

    :param D: A pandas dataframe of the predictor attributes and their values
    :param A: A dictionary where the keys are the attribute names and the values
        are lists of possible values
    :param C: A pandas series of values of the class attribute.
    :param num_attrs: Number of attributes to include in the sample
    :param num_obs: Number of rows to include in the sample
    :return: A tuple consisting of a pandas dataframe of the sample's predictor
        attributes/values, a pandas series of the sample's class attributes,
        and a dictionary of where the keys are the sample's attribute names
        and the values are lists of possible values
    """
    DC = D.copy()
    DC['class'] = C
    A_rand_keys = pd.Series(A.keys()).sample(n=num_attrs).to_list()
    A_rand = {a: A[a] for a in A_rand_keys}
    DC_rand = DC.sample(n=num_obs, replace=True)[A_rand_keys + ['class']]
    return DC_rand[A_rand_keys], DC_rand['class'], A_rand


def random_forest(corp: pd.Series,
                  num_attrs: int, num_obs: int, num_trees: int,
                  threshold: float = 0, gratio: bool = False) -> List[dict]:
    """
    Constructs random forest classifiers for a corpus of documents.

    :param corp: A pandas series of Vector objects representing documents
    :param num_attrs: Number of attributes to use in each tree
    :param num_obs: Number of rows to use in each tree's training sample
    :param num_trees: Number of trees to create in random forest
    :param threshold: (Default 0) Threshold used to prune trees in C45 algorithm
        (should be kept as 0)
    :param gratio: Whether to use gain or gains ratio
    :return: A list of dictionaries representing a random forest classifer
    """
    D = corp.apply(lambda v: v.tf_idf)
    A = {a: list(D[a].unique()) for a in list(D.columns)}
    C = corp.apply(lambda v: v.author())

    rf_trees = []
    for i in range(num_trees):
        print(f"Constructing tree {i+1}... ({i+1}/{num_trees})")
        D_train, C_train, A_train = dataset_selection(D, A, C,
                                                      num_attrs, num_obs)
        tree = C45(D_train, A_train, C_train,
                             threshold, gratio=gratio)
        rf_trees.append(tree)

    return rf_trees


def rf_predict(corp: pd.Series, trees: List[dict]) -> pd.DataFrame:
    """
    Uses random forest classfier to create authorship predictions for a dataset

    :param corp: A pandas series of Vector objects representing documents
    :param trees: List of dictionaries representing the random forest classifier
    :return: Original dataframe enriched with random forest predictions
    """
    df = corp.apply(lambda v: v.tf_idf)
    votes = {}
    for i in range(len(trees)):
        print(f"Classifying with tree {i+1}... ({i+1}/{len(trees)})")
        votes[i] = df.apply(
            lambda row: search_tree(row, trees[i]), axis=1)
    votes = pd.DataFrame(votes)

    results = pd.DataFrame()
    results['doc'] = corp.apply(lambda v: v.name())
    results['obs'] = corp.apply(lambda v: v.author())
    print("Voting...")
    results['pred'] = votes.apply(lambda row: row.mode().iloc[0], axis=1)
    return results


def main():
    args = sys.argv
    if len(args) < 6:
        print("Syntax: python3 rfAuthorship.py <VectorFile> "
              "<numTrees> <numAttrs> <numPts> <threshold>")

    else:
        vecfile = args[1]
        homedir = vecfile.split("_")[0]
        n_trees = int(args[2])
        n_attrs = int(args[3])
        n_pts = int(args[4])
        threshold = float(args[5])

        print(f"Reading vector file...")
        f = pd.read_csv(vecfile, index_col=0)
        print(f"Constructing vectors...")
        corp = vectors_from_f(f)

        trees = random_forest(corp, n_attrs, n_pts, n_trees,
                              threshold=threshold)
        results = rf_predict(corp, trees)

        results.to_csv(f"{homedir}_rf_results.csv", index=False)
        accuracy = (results['obs'] == results['pred']).sum() / len(
            results.index)

        print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
