# CSC 466 Fall 2023
# Martin Hsu - mshsu@calpoly.edu
# Lana Huynh - lmhuynh@calpoly.edu

import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations, chain
from typing import Any, Iterable, Optional


class AssocRule:
    """
    An object class representing an association rule X ---> Y
    """
    def __init__(self, x: Any, y: Any,
                 T: pd.Series, I: Optional[pd.Series] = None):
        # x is the left hand of the association rule
        self.x = x
        # y is the right hand of the association rule
        self.y = y
        # T is the set of market baskets. It should be a pandas series of lists.
        self.T = T
        # I is a series representing the names of the items. It is optional, but
        #   if it exists it should be included. This is not the same of the IDs.
        #   The IDs are the numbers, the names are the strings.
        self.I = I

    def __repr__(self):
        """
        String representation of class AssocRule when repr() method is called.
        Includes the actual names of the items, along with confidence and
        support.

        :return: String representation of class AssocRule
        """
        # This part checks if there are names included for the items.
        #   If there are names, you should include it.
        if self.I is not None:
            # Since some things (like author names) include commas, we separate
            #   the items using |. The lambda function here replaces the IDs
            #   with the actual string label.
            x = " | ".join(
                list(pd.Series(self.x).apply(lambda i: self.I.loc[i]))
            )
            y = " | ".join(
                list(pd.Series(self.y).apply(lambda i: self.I.loc[i]))
            )
        else:
            # This is if there are no names, then it just uses the IDs.
            x = ", ".join(list(pd.Series(self.x).astype(str)))
            y = ", ".join(list(pd.Series(self.y).astype(str)))
        sup = self.support()
        conf = self.confidence()
        return "%s ---> %s [sup = %.5f, conf = %.5f]" % (x, y, sup, conf)

    def __str__(self):
        """
        String representation of class AssocRule when str() method is called.
        Only includes IDs of items.

        :return: String representation of class AssocRule
        """
        # This step cleans up the string representation of the x and y a little
        x = ", ".join(list(pd.Series(self.x).astype(str)))
        y = ", ".join(list(pd.Series(self.y).astype(str)))
        return "%s -> %s" % (x, y)

    def confidence(self) -> float:
        """
        Calculates the confidence for the AssocRule.

        :return: A float representing the confidence of the AssocRule.
        """
        # This step turns x and y into sets. The if-else block checks if the
        #   element is an iterable or not, because the process for turning an
        #   iterable versus an atomic item is different.
        if isinstance(self.x, Iterable):
            x = set(self.x)
        else:
            x = {self.x}

        if isinstance(self.y, Iterable):
            y = set(self.y)
        else:
            y = {self.y}

        # This calls the confidence function. See the functions below.
        return confidence(x, y, self.T)

    def support(self, rel: bool = True) -> float:
        """
        Calculates the support for the itemset corresponding to the
        AssocRule.

        :param rel: (default True) When true, the method calculates relative
            support. When false, the method calculates absolute support.
        :return: A float representing the support of the itemset corresponding
            to the AssocRule.
        """
        # This step turns x and y into sets. The if-else block checks if the
        #   element is an iterable or not, because the process for turning an
        #   iterable versus an atomic item is different.
        if isinstance(self.x, Iterable):
            x = set(self.x)
        else:
            x = {self.x}

        if isinstance(self.y, Iterable):
            y = set(self.y)
        else:
            y = {self.y}

        # This calls the support function. See the functions below.
        if rel:
            return support(x.union(y), self.T) / len(self.T)
        else:
            return support(x.union(y), self.T)

    def itemset(self) -> tuple:
        """
        Returns the itemset corresponding to the AssocRule.

        :return: A tuple representing the itemset corresponding the AssocRule.
        """
        # This step turns x and y into sets. The if-else block checks if the
        #   element is an iterable or not, because the process for turning an
        #   iterable versus an atomic item is different.
        if isinstance(self.x, Iterable):
            x = set(self.x)
        else:
            x = {self.x}

        if isinstance(self.y, Iterable):
            y = set(self.y)
        else:
            y = {self.y}

        # Returns the union of left and right hand side of AssocRule as a tuple.
        return tuple(sorted(x.union(y)))


def read_sparse(filepath: str,
                Irm: Optional[Iterable[int]] = tuple()) -> pd.Series:
    """
    Takes a CSV filepath containing a sparse dataset and returns a pandas series
    of tuples representing the sparse dataset.

    :param filepath: A string representing a CSV filepath.
    :param Irm: A list of ID numbers to remove from the sets.
    :return: A pandas series of tuples representing a sparse dataset.
    """
    # Read in the data with pandas. Because sep="|" and we are assuming this
    #   is a CSV file, it will separate by line but not by column.
    sparse = pd.read_csv(filepath, sep="|", header=None)
    # At this point we have a dataset of strings. We turn these strings into
    #   lists representing the market baskets.
    # The list comprehension starts at 1 (x[1:]) because we assume the first
    # number is an index.
    sparse = sparse[0].str.split(", ") \
        .apply(lambda x: [int(i) for i in x[1:] if int(i) not in Irm])

    return sparse


def read_dense(filepath: str, Ipath: Optional[str] = None,
               Icol: Optional[Any] = 0, Isep: str = ",",
               Irm: Optional[Iterable[int]] = tuple()) -> pd.Series:
    """
    Takes a CSV filepath containing a dense dataset and returns a pandas series
    of tuples representing the sparse dataset (not dense!)

    :param filepath: A string representing a CSV filepath.
    :param Ipath: A string representing a CSV filepath. This should be the path
        to a dataset containing IDs of the items in the frequent itemsets. This
        is optional but should always be included if it exists, or there will
        be errors down the line.
    :param Icol: An int or string representing the column in the Ipath dataset
        that contains the ID numbers.
    :param Isep: A character representing the separator for the Ipath dataset
        that contains the ID numbers.
    :param Irm: A list of ID numbers to remove from the sets.
    :return: A pandas series of tuples representing a sparse dataset.
    """
    # Read in the dense dataset. We assume there is an index but no header.
    dense = pd.read_csv(filepath, index_col=0, header=None)\
        .reset_index(drop=True).T.reset_index(drop=True).T

    # If there is a path to the item IDs and names I...
    if Ipath is not None:
        I = pd.Series()
        # If Icol (the name of the column that contains ID numbers) is a number
        #   we do this...
        if isinstance(Icol, int):
            I = pd.read_csv(Ipath, sep=Isep).iloc[:, Icol]
        # If Icol is a string we do this...
        elif isinstance(Icol, str):
            I = pd.read_csv(Ipath, sep=Isep)[Icol]
        # We set the column names to the correct ID numbers.
        dense.columns = I

    # We convert the dense dataset to a sparse dataset by multiplying the
    #   market basket of 1s and 0s. by the ID numbers. We then remove the 0s,
    #   and then turn the market basket into a list instead of a bunch of column
    #   elements.
    sparse = pd.Series(
        dense.apply(lambda x: x * dense.columns, axis=1).values.tolist()
    ).apply(lambda x: [i for i in x if i != 0 and i not in Irm])

    return sparse


def support(x: set[Any], T: pd.Series, rel: bool = False) -> float:
    """
     Calculates the support given an itemset and the set of market baskets.

    :param x: A python set representing an itemset.
    :param T: A series of tuples representing a set of market baskets.
    :param rel: (default False) When true, the method calculates relative
        support. When false, the method calculates absolute support.
    :return: A float representing the support of the itemset
    """
    if rel:
        # If rel, calculate relative support
        # This takes advantage of python set operation subset. It checks each
        #   entry in the series, if x is a subset of the market basket then it
        #   is summed over.
        return T.apply(lambda l: x <= set(l)).sum() / len(T)
    else:
        # Else calculate absolute support in a similar way, just no denominator
        return T.apply(lambda l: x <= set(l)).sum()


def confidence(x: set[Any], y: set[Any], T: pd.Series) -> float:
    """
    Calculates the confidence of an association rule given the left and right
    hand sides and the market baskets.

    :param x: A python set representing the left hand side of an association
        rule
    :param y: A python set representing the right hand side of an association
        rule
    :param T: A series of tuples representing a set of market baskets.
    :return: A float representing the confidence of the association rule
    """
    # Equation: support(x and y) / support(x)
    return support(x.union(y), T) / support(x, T)


def first_pass(T: pd.Series, I: pd.Series, minSup: float) -> list[tuple[Any]]:
    """
    Performs the first pass portion of the apriori algorithm.

    :param T: A series of tuples representing the set of market baskets.
    :param I: A series of ID numbers representing the unique items.
    :param minSup: The minimum support to filter for.
    :return: A list of tuples representing the singleton itemsets that meet
        the minimum support condition.
    """
    # Calculate the support for each unique singleton item.
    supp = I.apply(lambda x: support({x}, T) / len(T))
    # Set the index to the correct ID numbers.
    supp.index = I
    # Turns the collection of items that meet minSup into a list of tuples.
    F_1 = [(i,) for i in I if supp.loc[i] > minSup]

    return F_1


def name_freq_itemsets(F: list[tuple[Any]],
                       Inames: pd.Series) -> list[tuple[Any]]:
    """
    Turns the frequent itemsets from a set of numeric IDs to a set of their
    actual string names.

    :param F: A list of tuples representing a set of itemsets, where the items
        are represented by ID numbers.
    :param Inames: A pandas series of strings representing the names of the
        items. The index should be the ID numbers of the strings.
    :return: A list of tuples representing a set of itemsets, where the items
        are represented by string names (eg. author name or food name).
    """
    # Convert pandas series to list
    return list(
        # Use pandas series capability
        pd.Series(F).apply(
            # This string of nested lambdas  first takes the list of items in
            #   in the itemsets, turn them into pandas series, turns each
            #   item from an ID number to a string name, then turns it into a
            #   tuple.
            lambda x: tuple(pd.Series(x).apply(lambda i: Inames.loc[i]))
        )
    )


def candidate_gen(F: list[tuple[Any]],
                  k: int) -> dict[tuple[Any], list[tuple[Any]]]:
    """
    Generates candidate itemsets of size k + 1 given a set of itemsets of size
    k.

    :param F: A list of tuples representing a set of itemsets of size k.
    :param k: An int representing the size of the tuples in F.
    :return: A dictionary, where the keys are tuples representing a set of
        itemsets of size k + 1, and the values are all its subsets of size k.
    """
    # Initialize the candidates as a dictionary.
    # This dictionary will have the candidate itemsets of size k+1 as the keys,
    #   and the subsets of size k as the values. This is so we can delete
    #   the subsets if the candidate is accepted, to get a skyline of frequent
    #   itemsets. For example: {(1, 2, 3): [(1, 2), (1, 3), (2, 3)]}
    C = {}
    # We use itertools.combinations() to get all combinations of 2 itemsets
    F_combs = list(combinations(F, 2))

    # This is now the standard algorithm as specified by Dr. Dekhtyar's
    #   pseudocode.
    for (f1, f2) in F_combs:
        c = list(set(f1 + f2))
        c.sort()
        if (len(f1) == len(f2)) and (len(f1) == k) and (len(c) == k + 1):
            flag = True
            S = list(combinations(c, k))
            for s in S:
                if s not in F:
                    flag = False
                    break
            tup_c = tuple(c)
            if flag:
                if tup_c not in C.keys():
                    C[tup_c] = S

    return C


def apriori(T: pd.Series, I: pd.Series,
            minSup: float, skyline: bool = True) -> list[tuple[Any]]:
    """
    Implements the apriori algorithm on a market basket to mine for frequent
    itemsets.

    :param T: A pandas series of tuples representing a set of market baskets.
    :param I: A pandas series of ID numbers.
    :param minSup: The minimum support to filter for.
    :param skyline: (Default True) When True, returns the skyline itemsets only.
        When False, returns all itemsets that exceed minSup.
    :return: A list of tuples representing the frequent itemsets that meet
        the minimum support condition.
    """
    # We initialize the frequent itemsets F as a dictionary. Each key will be
    #   an int representing itemset size, and the values are the itemsets of
    #   that size. For example: F[1] will contain freq itemsets of size 1.
    F = {}
    # Similar to F, but these are the candidate itemsets.
    C = {}

    # Set F[1] equal to output of first pass function - that is, the singleton
    #   frequent itemsets that pass minSup.
    F[1] = first_pass(T, I, minSup)
    # The rest is the standard algorithm as specified by Dr. Dekhtyar's
    #   pseudocode.
    k = 2

    while len(F[k - 1]) >= k:
        cand = candidate_gen(F[k - 1], k - 1)
        C[k] = cand.keys()
        count = {}
        for c in C[k]:
            count[c] = support(set(c), T)
        F[k] = [c for c in C[k] if count[c] / len(T) >= minSup]
        # This part deletes subsets of skyline itemsets.
        if skyline:
            for c in F[k]:
                for s in cand[c]:
                    if s in F[k - 1]:
                        F[k - 1].remove(s)
        k += 1

    # This uses the itertools.chain() function to collapse the dictionary F
    #   into a single list of frequent itemsets. It also deletes empty entries.
    return list(chain(*[val for key, val in F.items() if not len(val) == 0]))


def gen_rules(F: list[tuple[Any]], T: pd.Series,
              minConf: float, I: pd.Series = None) -> list[AssocRule]:
    """
    Generates association rules based on a set of frequent itemsets that meet
    a minimum confidence value.

    :param F: A list of tuples representing frequent itemsets identified by the
        apriori algorithm.
    :param T: A pandas series of tuples representing the set of market baskets.
    :param minConf: The minimum confidence to filter for.
    :param I: A pandas series of ID numbers.
    :return: A list of AssocRule class objects that meet the minimum confidence.
    """
    # This is based off the standard pseudocode provided by Dr. Dekhtyar.
    H1 = []
    for f in F:
        if len(f) >= 2:
            for s in f:
                x = set(f)
                x.remove(s)
                if confidence(x, {s}, T) >= minConf:
                    # We use the AssocRule object that we created here.
                    H1.append(AssocRule(tuple(x), s, T, I))
    return H1

# THIS IS THE MAIN FUNCTION TO USE TO RUN ALL APRIORI STEPS!!!!!!
def apriori_full(Tpath: str, minSup: float, minConf: float,
                 Ipath: Optional[str] = None, Icol: Optional[Any] = 0,
                 Inames: Optional[Any] = None, Isep: str = ",",
                 Irm: Optional[Iterable[int]] = tuple(),
                 outpath: Optional[str] = None, show: bool = True,
                 arules: bool = True, fsets: bool = True,
                 sparse: bool = True, skyline: bool = True
                 ) -> tuple[list[tuple], list[AssocRule]]:
    """
    Runs all steps of the apriori process association rules mining process
    given filepaths or links to the market baskets, minimum support and minimum
    confidence.

    :param Tpath: A string filepath or URL to a CSV file that contains the
        set of market baskets, T.
    :param minSup: A float representing the minimum support to filter itemsets
        for.
    :param minConf: A float representing the minimum confidence to filter
        association rules for.
    :param Ipath: (Default None) A string filepath or URL to a delimited file
        that contains the ID numbers and names for all the unique items in the
        market baskets. This is optional, but should always be included if it
        exists.
    :param Icol: (Default 0) A string or int representing the column in the
        Ipath dataset to reference for ID numbers. This is optional, but should
        always be included if it exists.
    :param Inames: (Default None) A string or int, or a list of strings or ints
        representing the column in the Ipath dataset that contains the names
        for all the unique items in the market baskets. This is optional, but
        should always be included if it exists.
    :param Isep: (Default ",") A string representing the delimiter for the Ipath
        file. This should be specified if Ipath exists and is not a CSV file.
    :param Irm: (Default empty tuple) A list of item ID numbers to remove from
        all market baskets.
    :param outpath: (Default None) A string representing a txt filepath to
        which to output the printed results of the apriori run. Will not output
        if value is None.
    :param show: (Default True) If True, will print results of apriori run to
        the console. Will not print if False.
    :param arules: (Default True) If True, will print association rules mined by
        the apriori run to the console. Will not print if False, even if `show`
        is True.
    :param fsets: (Default True) If True, will print frequent itemsets mined by
        the apriori run to the console. Will not print if False, even if `show`
        is True.
    :param sparse: (Default True) If True, assumes the dataset specified in
        Tpath is a sparse dataset. If False, assumes it is dense.
    :param skyline: (Default True) If True, will print only skyline frequent
        itemsets mined by apriori run to the console. If False, will print
        all frequent itemsets mined by the run to the console.
    :return: A tuple where the first element is a list of tuples representing
        frequent itemsets that meet the minimum support, and the second element
        is a list of AssocRule objects representing the association rules that
        meet the minimum confidence that correspond to the frequent itemsets.
    """
    # Loads market baskets data differently depending on if data is sparse or
    #   not.
    # This means the function can handle both sparse or dense data!!!
    if sparse:
        T = read_sparse(Tpath, Irm=Irm)
    else:
        T = read_dense(Tpath, Ipath, Icol=Icol, Isep=Isep, Irm=Irm)

    # Loads ID and names of the items, if they exist
    I = pd.Series(range(len(T)))
    names = None
    if Ipath is not None:
        # This part reads in the ID numbers
        if isinstance(Icol, int):
            I = pd.read_csv(Ipath, sep=Isep).iloc[:, Icol]
        elif isinstance(Icol, str):
            I = pd.read_csv(Ipath, sep=Isep)[Icol]

        # This part reads in the names, if they exist.
        if Inames is not None:
            if isinstance(Inames, int):
                names = pd.read_csv(Ipath, sep=Isep).iloc[:, Inames].str.strip()
            elif isinstance(Inames, str):
                names = pd.read_csv(Ipath, sep=Isep)[Inames].str.strip()
            else:
                names = pd.read_csv(Ipath, sep=Isep)[Inames].agg(' '.join,
                                                                 axis=1)

    # Runs the apriori algorithm for frequent itemsets, stores output.
    F = apriori(T, I, minSup, skyline)
    # This part applies the string names to the ID numbers if they exist.
    if (names is not None) and (Ipath is not None):
        F_named = name_freq_itemsets(F, names)
    else:
        F_named = F

    # Runs algorithm for association rules, stores output.
    assrules = gen_rules(F, T, minConf, names)

    # This part formats and prints the results to the console, based on user
    #   specifications.
    if show:
        print("minSup: %f, minConf: %f" % (minSup, minConf))
        if fsets:
            if skyline:
                print()
                print("Skyline Frequent Itemsets:")
                print(F_named)
            else:
                print("All Frequent Itemsets:")
                print(F_named)

        if arules:
            print()
            print("Singleton Association Rules:")
            for i in range(len(assrules)):
                print("Rule %d: %s" % (i + 1, repr(assrules[i])))

    # This part formats and prints the results to a specified TXT file, based
    #   on user specifications.
    if outpath is not None:
        file = open(outpath, "w")
        file.write("minSup: %f, minConf: %f" % (minSup, minConf))
        if fsets:
            if skyline:
                file.write("\n\nSkyline Frequent Itemsets\n")
                file.write(str(F_named))
            else:
                file.write("\n\nAll Frequent Itemsets\n")
                file.write(str(F_named))

        if arules:
            file.write("\n\nSingleton Association Rules:")
            for i in range(len(assrules)):
                file.write("\nRule %d: %s" % (i + 1, repr(assrules[i])))
        file.close()

    # Returns the frequent itemsets and association rules incase they need to be
    #   used later
    return F, assrules


def apriori_explore(Tpath, minSup, type: Optional[str] = "all",
                    Ipath: Optional[str] = None, Icol: Optional[Any] = 0,
                    Isep: str = ",", Irm: Optional[Iterable[int]] = tuple(),
                    skyline: bool = True, sparse: bool = True) -> None:
    """
    Returns visualizations based on results from the apriori algorithm.

    :param Tpath: A string filepath or URL to a CSV file that contains the
        set of market baskets, T.
    :param minSup: A float representing the minimum support to filter itemsets
        for.
    :param type: (Default "all") If "support", only returns support chart. If
        "confidence," only returns confidence chart. If "all" or None, returns
        both.
    :param Ipath: (Default None) A string filepath or URL to a delimited file
        that contains the ID numbers and names for all the unique items in the
        market baskets. This is optional, but should always be included if it
        exists.
    :param Icol: (Default 0) A string or int representing the column in the
        Ipath dataset to reference for ID numbers. This is optional, but should
        always be included if it exists.
    :param Isep: (Default ",") A string representing the delimiter for the Ipath
        file. This should be specified if Ipath exists and is not a CSV file.
    :param Irm: A list of ID numbers to remove from the sets.
    :param skyline: (Default True) If True, will include only skyline frequent
        itemsets mined by apriori in visualizations. If False, will print
        all itemsets mine by apriori in visualizations.
    :param sparse: (Default True) If True, assumes the dataset specified in
        Tpath is a sparse dataset. If False, assumes it is dense.
    :return: None
    """
    # Does the apriori full run
    F, R = apriori_full(Tpath, minSup, 0, show=False,
                        Ipath=Ipath, Icol = Icol, Isep = Isep, Irm = Irm,
                        skyline=skyline, sparse=sparse)

    # Calculate confidence for all association rules
    conf = pd.Series(R).apply(lambda x: x.confidence())
    conf.index = pd.Series(R)

    # Calculate support for all itemsets of size >=2
    supp = pd.DataFrame({
        "supp": pd.Series(R).apply(lambda x: x.support()),
        "fsets": pd.Series(R).apply(lambda x: x.itemset())
    }).drop_duplicates().set_index("fsets")['supp']
    supp.index.name = None

    # Plot the charts based on input preferences
    if type == "confidence":
        conf.sort_values().plot.bar()
        plt.title("Confidence")
        plt.show()
    elif type == "support":
        supp.sort_values().plot.bar()
        plt.title("Support")
        plt.show()
    elif type is None or type == "all":
        supp.sort_values().plot.bar()
        plt.title("Support")
        plt.show()

        conf.sort_values().plot.bar()
        plt.title("Confidence")
        plt.show()
    else:
        plt.show()

