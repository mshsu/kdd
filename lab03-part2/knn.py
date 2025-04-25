# Martin Hsu - mshsu@calpoly.edu
# Lana Huynh - lmhuynh@calpoly.edu

import classify

import pandas as pd
import numpy as np
import sys


def dummify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dummifies categorical variables in a pandas dataframe.

    :param df: A pandas dataframe
    :return: Same pandas dataframe with categorical variables dummified
    """
    df_dummy = df.copy()
    cat_cols = [col for col in df.columns
                if pd.api.types.is_object_dtype(df[col])]

    new_dummies = {}
    for col in cat_cols:
        cats = set(df[col])
        for cat in cats:
            new_dummies[col + '_' + str(cat)] = (df[col] == cat) * 1
        df_dummy.drop(col, axis=1, inplace=True)
    df_dummy = pd.concat([df_dummy, pd.DataFrame(new_dummies)], axis=1)

    return df_dummy


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes numeric data in pandas dataframe.

    :param df: A pandas dataframe
    :return: Same pandas dataframe with numeric variables standardized
        (centered and normalized)
    """
    df_std = df.copy()
    num_cols = [col for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col])]

    for col in num_cols:
        df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()

    return df_std


def knn(row: pd.Series, D: pd.DataFrame, C: pd.Series, k: int) -> str:
    """
    Finds the most frequent class attribute label from an observation's
    k-nearest neighbors in D

    :param row: A pandas series representing a single row conforming to the
        schema in D
    :param D: A pandas dataframe of data observations
    :param C: A pandas series representing the class attribute labels of the
        rows in D
    :param k: Number of nearest neighbors to evaluate
    :return: Most frequent class attribute label from row's k-nearest neighbors
        in D
    """
    dists = np.sqrt(((D - row)**2).sum(axis=1))
    # np.argpartition will only partially sort the indices of the k smallest
    #   distances into the first k positions in the list. That way we don't
    #   waste time sorting the entire dataset for every single fkn row >:)
    # This change slashed the runtime down to a fraction of what it was before!
    top_k_dist_ind = np.argpartition(dists, k)[:k]
    top_k_c = C.iloc[top_k_dist_ind]
    return top_k_c.mode().sample(frac=1).iloc[0]


def knn_predict(D: pd.DataFrame, k: int, classvar: str,
                restrict: bool = None) -> pd.DataFrame:
    """
    Runs KNN on each row of a dataframe, and outputs the dataframe with the
        predicted values.

    :param D: A pandas series representing a single row conforming to the
        schema in D
    :param k: Number of nearest neighbors to evaluate
    :param classvar: category attribute to be classified
    :param restrict: An optional binary list indicating which attributes to use
        to induce KNN
    :return: original pandas dataframe enriched with KNN predictions
    """
    cols = list(D.columns)
    if restrict is not None:
        cols = [cols[i] for i in range(len(cols)) if restrict[i] == 1]

    D_orig = D.copy()
    C = D_orig[classvar]
    D = dummify(standardize(D[cols].drop(classvar, axis=1)))

    D_orig['pred'] = pd.Series(D.index).apply(
        lambda row: knn(D.loc[row], D.drop(row), C.drop(row), k)
    )
    return D_orig


def main():
    args = sys.argv

    # Handle customizations
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

    if len(args) < 3:
        print("Syntax: python3 knn.py <CSVFile.csv> <NumNeighbors> "
              "[<restrictionsFile>] [--nooutput] [--noprint]")
    else:
        path = args[1].strip("'").strip('"')
        k = int(args[2])

        filename = path.split('/')[-1]

        raw = pd.read_csv(path, skiprows=[1], header=0)
        classvar = raw.iloc[0, 0]
        data = pd.read_csv(path, skiprows=[1, 2], header=0)
        data[classvar] = data[classvar].astype(str)

        restrict = None
        if len(args) == 4:
            rfilepath = args[3]
            with open(rfilepath, 'r') as rfile:
                restrict = [int(i) for i in rfile.read().split(",")]

        pred = knn_predict(data, k, classvar, restrict=restrict)

        results = classify.report(pred, classvar, 'pred')
        results_table = pd.DataFrame()
        results_table['Overall Precision:'] = results['precision']
        results_table['Overall Recall:'] = results['recall']
        results_table.index.name = None
        results_table = results_table.T

        report_text = (f"Martin Hsu - mshsu@calpoly.edu\n"
                       f"Lana Huynh - lmhuynh@calpoly.edu\n\n"
                       f"Data: {filename}\nClass Attribute: {classvar}"
                       f"\n\nConfusion Matrix:"
                       f"\n" + results['conf_matrix'].to_string() +
                       f"\n\nRecords Classified: {results['n']}"
                       f"\nRecords Correctly Classified: "
                       f"{results['n_correct']}"
                       f"\nRecords Incorrectly Classified: "
                       f"{results['n_incorrect']}"
                       f"\nOverall Accuracy: {results['accuracy']:0.3f}"
                       f"\nOverall Error Rate: "
                       f"{results['error_rate']:0.3f}\n"
                       + results_table.to_string())

        if printout:
            print(report_text)

        if output:
            prefix = filename.replace(".csv", "")
            reportpath = prefix + "_knn_results.out"
            with open(reportpath, 'w') as report_file:
                report_file.write(report_text)


if __name__ == "__main__":
    main()
