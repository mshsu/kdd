# Martin Hsu - mshsu@calpoly.edu
# Lana Huynh - lmhuynh@calpoly.edu

import classify
import induceC45

import pandas as pd
import numpy as np
import sys

from typing import List, Optional


def split_folds(df: pd.DataFrame, k: int) -> List:
    """
    Splits our file of records dataframe into k folds.

    :param df: pandas dataframe of file of records
    :param k: number of folds to be split into, with k = 0 being no
        cross-validation and k = -1 representing all-but-one cross
        validation
    :return: list of k random subsets of our original dataframe
    """
    return np.array_split(df.sample(frac=1), k)


def cv_report(splits: List[pd.DataFrame], classvar: str,
              threshold: float, gratio: bool = False,
              restrict: Optional[List[int]] = None) -> dict:
    """
    Performs a cross-validation analysis and produces a report based
      on our output.

    :param splits: evenly divided subsets of original file of records
    :param classvar: category attribute to be classified
    :param threshold: splitting value for atrribute values to be 
        classified
    :param gratio: if True use information gain ratio, 
        if False use information gain
    :param restrict: An optional binary list indicating which attributes to use
        to induce the decision tree
    :return: dictionary of reports of all folds
    """
    preds = []
    results = []

    for i in range(len(splits)):
        # Take fold i as test data
        test = splits[i].copy()

        # Take rest of folds, combine, and fit tree.
        train = splits[:]
        del train[i]
        train = pd.concat(train)

        C_train = train[classvar]
        D_train = train.drop(classvar, axis=1)
        A_train = {a: list(D_train[a].unique()) for a in list(D_train.columns)}
        if restrict is not None:
            A_remove = [list(A_train.keys()) for i in range(len(A_train))
                        if restrict[i] == 0]
            for a in A_remove:
                del A_train[a]

        T = induceC45.C45(D_train, A_train, C_train, threshold, gratio=gratio)

        pred = classify.predict(test, T)
        preds.append(pred)
        results.append(classify.report(pred, classvar, 'pred'))

    preds = pd.concat(preds)
    results = pd.DataFrame(results)
    cv_report = classify.report(preds, classvar, 'pred')
    cv_report['avg_accuracy'] = results['accuracy'].mean()
    return cv_report


def fit_loo(i: int, df: pd.DataFrame, classvar: str,
            threshold: float, gratio: bool = False,
            restrict: Optional[List[int]] = None) -> str:
    """
    Helper function to apply leave one out cross-validation

    :param i: index in datafreame to leave out
    :param df: CSV file of records to be classified
    :param classvar: category attribute to be classified
    :param threshold: splitting value for atrribute values to be 
        classified
    :param gratio: if True use information gain ratio, 
        if False use information gain
    :param restrict: An optional binary list indicating which attributes to use
        to induce the decision tree
    :return: predicted classification of a single row in dataframe
    """
    loo = df.drop(i, axis=0)
    C = loo[classvar]
    D = loo.drop(classvar, axis=1)
    A = {a: list(D[a].unique()) for a in list(D.columns)}
    if restrict is not None:
        A_remove = [list(A.keys()) for i in range(len(A))
                    if restrict[i] == 0]
        for a in A_remove:
            del A[a]

    T = induceC45.C45(D, A, C, threshold, gratio=gratio)

    row = df.loc[i].to_frame().T
    pred = classify.predict(row, T)['pred'].loc[i]
    return pred


def loo_predict(df: pd.DataFrame, classvar: str,
                threshold: float, gratio: bool = False,
                restrict: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Applies all-but-one cross-validation to entire dataframe using 
        vectorization.

    :param df: CSV file of records to be classified
    :param classvar: category attribute to be classified
    :param threshold: splitting value for atrribute values to be 
        classified
    :param gratio: if True use information gain ratio, 
        if False use information gain
    :param restrict: An optional binary list indicating which attributes to use
        to induce the decision tree
    :return: new dataframe containing original file of records and 
        predicted values
    """
    pred = pd.Series(df.index).apply(lambda i: fit_loo(i, df, classvar,
                                                       threshold, gratio=gratio,
                                                       restrict=restrict))
    df['pred'] = pred
    return df


def main():
    args = sys.argv

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

    if len(args) < 4:
        print("Syntax: python3 validation.py <TrainingSetFile.csv> <threshold> "
              "<nfolds> [<restrictionsFile>] [--gratio] [--nooutput] "
              "[--noprint]")
    else:
        path = args[1].strip("'").strip('"')
        threshold = float(args[2])
        k = int(args[3])

        filename = path.split('/')[-1]

        raw = pd.read_csv(path, skiprows=[1], header=0)
        classvar = raw.iloc[0, 0]
        data = pd.read_csv(path, skiprows=[1, 2], header=0)
        data[classvar] = data[classvar].astype(str)

        report_text = ""

        restrict = None
        if len(args) == 5:
            rfilepath = args[4]
            with open(rfilepath, 'r') as rfile:
                restrict = [int(i) for i in rfile.read().split(",")]

        if k >= 2:
            splits = split_folds(data, k)
            results = cv_report(splits, classvar, threshold,
                                gratio=gratio, restrict=restrict)
            results_table = pd.DataFrame()
            results_table['Overall Precision:'] = results['precision']
            results_table['Overall Recall:'] = results['recall']
            results_table.index.name = None
            results_table = results_table.T

            report_text = (f"Martin Hsu - mshsu@calpoly.edu\n"
                           f"Lana Huynh - lmhuynh@calpoly.edu\n\n"
                           f"Data: {filename}\nClass Attribute: {classvar}"
                           f"\nValidation: {k}-Fold Cross Validation\n"
                           f"\nConfusion Matrix:"
                           f"\n" + results['conf_matrix'].to_string() +
                           f"\n\nRecords Classified: {results['n']}"
                           f"\nRecords Correctly Classified: "
                           f"{results['n_correct']}"
                           f"\nRecords Incorrectly Classified: "
                           f"{results['n_incorrect']}"
                           f"\nOverall Accuracy: {results['accuracy']:0.3f}"
                           f"\nAverage Accuracy: {results['avg_accuracy']:0.3f}"
                           f"\nOverall Error Rate: "
                           f"{results['error_rate']:0.3f}\n"
                           + results_table.to_string())
        elif k == -1:
            pred = loo_predict(data, classvar, threshold,
                               gratio=gratio, restrict=restrict)
            results = classify.report(pred, classvar, 'pred')
            results_table = pd.DataFrame()
            results_table['Overall Precision:'] = results['precision']
            results_table['Overall Recall:'] = results['recall']
            results_table.index.name = None
            results_table = results_table.T

            report_text = (f"Martin Hsu - mshsu@calpoly.edu\n"
                           f"Lana Huynh - lmhuynh@calpoly.edu\n\n"
                           f"Data: {filename}\nClass Attribute: {classvar}"
                           f"\nValidation: Leave-One-Out\n\nConfusion Matrix:"
                           f"\n" + results['conf_matrix'].to_string() +
                           f"\n\nRecords Classified: {results['n']}"
                           f"\nRecords Correctly Classified: "
                           f"{results['n_correct']}"
                           f"\nRecords Incorrectly Classified: "
                           f"{results['n_incorrect']}"
                           f"\nOverall Accuracy: {results['accuracy']:0.3f}"
                           f"\nOverall Error Rate: "
                           f"{results['error_rate']:0.3f}\n" +
                           results_table.to_string())

        elif k in (0, 1):
            C = data[classvar]
            D = data.drop(classvar, axis=1)
            A = {a: list(D[a].unique()) for a in list(D.columns)}
            if restrict is not None:
                A_remove = [list(A.keys()) for i in range(len(A))
                            if restrict[i] == 0]
                for a in A_remove:
                    del A[a]

            T = induceC45.C45(D, A, C, threshold, gratio=gratio)
            pred = classify.predict(data, T)
            results = classify.report(pred, classvar, "pred")
            results_table = pd.DataFrame()
            results_table['Overall Precision:'] = results['precision']
            results_table['Overall Recall:'] = results['recall']
            results_table.index.name = None
            results_table = results_table.T

            report_text = (f"Martin Hsu - mshsu@calpoly.edu\n"
                           f"Lana Huynh - lmhuynh@calpoly.edu\n\n"
                           f"Data: {filename}\n"
                           f"Class Attribute: {classvar}\nValidation: None\n"
                           f"\nConfusion Matrix:"
                           f"\n" + results['conf_matrix'].to_string() +
                           f"\n\nRecords Classified: {results['n']}"
                           f"\nRecords Correctly Classified: "
                           f"{results['n_correct']}"
                           f"\nRecords Incorrectly Classified: "
                           f"{results['n_incorrect']}"
                           f"\nOverall Accuracy: {results['accuracy']:0.3f}"
                           f"\nOverall Error Rate: "
                           f"{results['error_rate']:0.3f}\n" +
                           results_table.to_string())

        if printout:
            print(report_text)

        if output:
            prefix = filename.replace(".csv", "")
            reportpath = prefix + "_results.out"
            with open(reportpath, 'w') as report_file:
                report_file.write(report_text)


if __name__ == "__main__":
    main()
