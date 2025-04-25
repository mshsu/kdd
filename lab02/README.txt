# NAMES AND EMAILS
- Martin Hsu | mshsu@calpoly.edu
- Lana Huynh | lmhuynh@calpoly.edu

# SUBMISSION DESCRIPTION
Python Script (apriori.py)
Jupyter Notebook (lab02.ipynb)

# INSTRUCTIONS
Must have the following packages:
- matplotlib, pandas, itertools, typing

Structure: The code consists of two parts.
- apriori.py - contains all the functions and classes, the most important of
which is apriori_full(), which is the main function that runs everything and
outputs all the results and takes in input that conforms to Dr. Dekhtyar's
specifications.
- lab02.ipynb - loads the functions from apriori.py and uses them to perform
analyses. This notebook is where all the results are.

How to run:
1. Load the files into your environment of choice. Make sure that apriori.py and
lab02.ipynb are contained together in the same folder!!!! That way the notebook
can import and run the functions from apriori.py.
2. Open lab02.ipynb. Press "Run All" or any equivalent to run all code chunks.
3.

Functions:
For function documentation, please reference the annotations in
apriori.py!!! Every single function's purpose, parameters, parameter types and
what it returns, as well as specific notes for parts of the code, is in there.
These are the most important functions:
1. apriori_full() - runs all steps of the association rules mining process
according to lab specifications
2. apriori_explore() - outputs visualizations for association rules exploration
The functions containing the algorithms as specified by Dr. Dekhtyar are:
- apriori() - the apriori freq itemsets mining algorithm implemented based on
Dr. Dekhtyar's pseudocode
- candidate_gen() - the freq itemsets candidate generating algorithm implemented
based on Dr. Dekhtyar's pseudocode
- gen_rules() - the singleton association rules mining algorithm implemented
based on Dr. Dekhtyar's pseudocode

Inputs for apriori_full():
Description: Runs all steps of the apriori process association rules mining
process given filepaths or links to the market baskets, minimum support and
minimum confidence.
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
    :param Irm: A list of ID numbers to remove from the sets.
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

# PROGRAMS/FILES
- apriori.py
- lab02.ipynb
- Lab2-Report.pdf
- README.txt

# EXPECTED ERRORS
- You MUST have apriori.py and lab02.py in the SAME LOCATION or there will be
errors.
- If there is a file with IDs and names for the market basket, you must add it
to the function parameters, or you may get empty output or even errors.
- You must specify all parts of the function accurately!!! There are many
parameters; this is because each dataset is different and needs to be loaded
differently according to its structure (such as sparse vs dense, CSV vs PSV).
For example, if the market basket is a dense dataset but you set sparse=True
in the apriori_full() function, you WILL get an error.
As another example, if the dataset of items is "|" delimited (PSV) but you don't
set sep="|" to "|" in the apriori_full() function, you WILL get an error.
- We cannot stress this enough. We have every single function documented out
with every parameter explained and notes everywhere. Each parameter's data type
has been labeled. You SHOULD NOT get any errors if every parameter is specified
correctly, and it should be clear what each parameter is for.
- If you need an example on how to run things for the bakery or bingo datasets,
reference the lab02.ipynb file.