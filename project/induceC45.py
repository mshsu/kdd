# Martin Hsu - mshsu@calpoly.edu
# Lana Huynh - lmhuynh@calpoly.edu

import pandas as pd
import numpy as np
import sys
import json

from typing import List, Optional, Tuple


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


def main():
    # Read args from command line
    args = sys.argv[:]

    # Handle customizations
    # Use gains ratio if --gratio specified
    gratio = False
    if "--gratio" in args:
        gratio = True
        args.remove("--gratio")
    # Don't save tree to json file if --nooutput specified
    output = True
    if "--nooutput" in args:
        output = False
        args.remove("--nooutput")
    # Don't print tree to console if --noprint specified
    printout = True
    if "--noprint" in args:
        printout = False
        args.remove("--noprint")

    # If arguments entered incorrectly, remind user of syntax. Otherwise, run
    #   C45 algorithm
    if len(args) < 3:
        print("Syntax: python3 induceC45.py <TrainingSetFile.csv> <threshold> "
              "[<restrictionsFile>] [--gratio] [--nooutput] [--noprint]")
    else:
        path = args[1].strip("'").strip('"')
        threshold = float(args[2])

        filename = path.split('/')[-1]

        raw = pd.read_csv(path, skiprows=[1], header=0)
        classvar = raw.iloc[0, 0]
        data = pd.read_csv(path, skiprows=[1, 2], header=0)
        data[classvar] = data[classvar].astype(str)

        C = data[classvar]
        D = data.drop(classvar, axis=1)
        A = {a: list(D[a].unique()) for a in list(D.columns)}
        if len(args) == 4:
            rfilepath = args[3]
            with open(rfilepath, 'r') as rfile:
                restrict = [int(i) for i in rfile.read().split(",")]
            A_remove = [list(A.keys()) for i in range(len(A))
                        if restrict[i] == 0]
            for a in A_remove:
                del A[a]

        T = C45(D, A, C, threshold, gratio=gratio)
        T["dataset"] = filename

        json_data = json.dumps(T, indent=1)
        if printout:
            print(json_data)
        if output:
            outpath = filename.replace(".csv", "") + "_tree.json"
            with open(outpath, 'w') as json_file:
                json_file.write(json_data)


if __name__ == "__main__":
    main()
