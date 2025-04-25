# Martin Hsu - mshsu@calpoly.edu
# Lana Huynh - lmhuynh@calpoly.edu

import induceC45
import classify
import validation

import pandas as pd
import sys

from typing import List, Tuple, Optional


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


def random_forest(D: pd.DataFrame, A: dict, C: pd.Series,
                  num_attrs: int, num_obs: int, num_trees: int,
                  threshold: float = 0, gratio: bool = False) -> List[dict]:
    """
    Constructs random forest classifiere.

    :param D: A pandas dataframe of the predictor attributes and their values.
    :param A: A dictionary where the keys are the attribute names and the values
        are the possible values.
    :param C: A pandas series of values of the class attribute.
    :param num_attrs: Number of attributes to use in each tree
    :param num_obs: Number of rows to use in each tree's training sample
    :param num_trees: Number of trees to create in random forest
    :param threshold: (Default 0) Threshold used to prune trees in C45 algorithm
        (should be kept as 0)
    :param gratio: Whether to use gain or gains ratio
    :return: A list of dictionaries representing a random forest classifer
    """
    rf_trees = []
    for i in range(num_trees):
        D_train, C_train, A_train = dataset_selection(D, A, C,
                                                      num_attrs, num_obs)
        tree = induceC45.C45(D_train, A_train, C_train,
                             threshold, gratio=gratio)
        rf_trees.append(tree)

    return rf_trees


def rf_predict(df: pd.DataFrame, trees: List[dict]) -> pd.DataFrame:
    """
    Uses random forest classfier to create predictions for a dataset

    :param df: Pandas dataframe representing the data
    :param trees: List of dictionaries representing the random forest classifier
    :return: Original dataframe enriched with random forest predictions
    """
    votes = {}
    for i in range(len(trees)):
        votes[i] = df.apply(
            lambda row: classify.search_tree(row, trees[i]), axis=1)
    votes = pd.DataFrame(votes)
    df['pred'] = votes.apply(lambda row: row.mode().iloc[0], axis=1)
    return df


def rf_report(splits: List[pd.DataFrame], classvar: str,
              num_attrs: int, num_obs: int, num_trees: int,
              threshold: float = 0, gratio: bool = False,
              restrict: Optional[List[int]] = None) -> dict:
    """
    Performs a cross validation with random forest classifier and outputs
    results

    :param splits: evenly divided subsets of original file of records
    :param classvar: category attribute to be classified
    :param num_attrs: Number of attributes to use in each tree
    :param num_obs: Number of rows to use in each tree's training sample
    :param num_trees: Number of trees to create in random forest
    :param threshold: (Default 0) Threshold used to prune trees in C45 algorithm
        (should be kept as 0)
    :param gratio: Whether to use gain or gains ratio
    :param restrict: An optional binary list indicating which attributes to use
        to induce KNN
    :return: reports total number of records classified,
      correctly/incorrectly classified, overall accuracy and
      error rate
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
        trees = random_forest(D_train, A_train, C_train,
                              num_attrs, num_obs, num_trees,
                              threshold, gratio=gratio)
        pred = rf_predict(test, trees)
        preds.append(pred)
        results.append(classify.report(pred, classvar, 'pred'))

    preds = pd.concat(preds)
    results = pd.DataFrame(results)
    cv_report = classify.report(preds, classvar, 'pred')
    cv_report['avg_accuracy'] = results['accuracy'].mean()
    return cv_report


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

    if len(args) < 5:
        print("Syntax: python3 randomforest.py <TrainingSetFile.csv> "
              "<NumAttributes> <NumDataPoints> <NumTrees> "
              "[<restrictionsFile>] [--gratio] [--nooutput] "
              "[--noprint]")
    else:
        path = args[1].strip("'").strip('"')
        num_attrs = int(args[2])
        num_obs = int(args[3])
        num_trees = int(args[4])

        filename = path.split('/')[-1]

        raw = pd.read_csv(path, skiprows=[1], header=0)
        classvar = raw.iloc[0, 0]
        data = pd.read_csv(path, skiprows=[1, 2], header=0)
        data[classvar] = data[classvar].astype(str)

        restrict = None
        if len(args) == 6:
            rfilepath = args[4]
            with open(rfilepath, 'r') as rfile:
                restrict = [int(i) for i in rfile.read().split(",")]

        splits = validation.split_folds(data, 10)
        results = rf_report(splits, classvar,
                            num_attrs, num_obs, num_trees,
                            0, restrict=restrict, gratio=gratio)
        results_table = pd.DataFrame()
        results_table['Overall Precision:'] = results['precision']
        results_table['Overall Recall:'] = results['recall']
        results_table.index.name = None
        results_table = results_table.T

        report_text = (f"Martin Hsu - mshsu@calpoly.edu\n"
                       f"Lana Huynh - lmhuynh@calpoly.edu\n\n"
                       f"Data: {filename}\nClass Attribute: {classvar}"
                       f"\nValidation: 10-Fold Cross Validation\n"
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

        if printout:
            print(report_text)

        if output:
            prefix = filename.replace(".csv", "")
            reportpath = prefix + "_rf_results.out"
            with open(reportpath, 'w') as report_file:
                report_file.write(report_text)


if __name__ == "__main__":
    main()
