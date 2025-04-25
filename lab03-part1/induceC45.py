# Martin Hsu - mshsu@calpoly.edu
# Lana Huynh - lmhuynh@calpoly.edu

import pandas as pd
import numpy as np
import sys
import json

from typing import List, Optional, Tuple


def entropy(D: pd.DataFrame, C: pd.Series,
            A_i: Optional[str] = None, x: Optional[float] = None) -> float:
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
        # TODO: Implement this in part 2
        pass

    return e


# TODO: Implement this in part 2
def findBestSplit(A_i: str, D: pd.DataFrame) -> float:
    pass


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
                             gratio: bool = False) -> Optional[str]:
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
    p[0] = entropy(D, C)
    for A_i in A:
        # TODO: Implement this in part 2
        # if pd.api.types.is_numeric_dtype(D.dtypes[A_i]):
        #     x = findBestSplit(A_i, D)
        #     p[A_i] = entropy(D, C, A_i, x)
        # else:
        #     p[A_i] = entropy(D, C, A_i)
        # TODO: Delete next line once part 2 implemented
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
        return best
    else:
        return None


def C45(D: pd.DataFrame, A: List[str], C: pd.Series,
        threshold: float, gratio: bool = False) -> dict:
    """
    Implements the C45 algorithm to construct a decision tree classifier.

    :param D: A list of attribute names.
    :param A: A pandas dataframe of the predictor attributes and their values.
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
        A_g = selectSplittingAttribute(A, D, C, threshold, gratio=gratio)
        if A_g is None:
            c, p = find_most_frequent_label(C)
            T["leaf"] = {"decision": c, "p": p}
        else:
            r = {"var": A_g, "edges": []}
            T["node"] = r
            for v in list(D[A_g].unique()):
                D_v = D[D[A_g] == v]
                C_v = C[D[A_g] == v]
                if len(D_v.index) != 0:
                    # TODO: Implement this in part 2
                    # if D.dtypes[A_g] == object:
                    #     A_v = A[:]
                    #     A_v.remove(A_g)
                    # TODO: Delete next 2 lines once part 2 implemented
                    A_v = A[:]
                    A_v.remove(A_g)
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
        data = raw.drop(0, axis=0)

        C = data[classvar]
        D = data.drop(classvar, axis=1)
        A = list(D.columns)
        if len(args) == 4:
            rfilepath = args[3]
            with open(rfilepath, 'r') as rfile:
                restrict = [int(i) for i in rfile.read().split(",")]
            A = [A[i] for i in range(len(A)) if restrict[i] == 1]

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
