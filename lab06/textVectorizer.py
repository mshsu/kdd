import pandas as pd
import numpy as np
import os
import sys

from nltk.stem.porter import PorterStemmer
from typing import Optional


default_stopwords_link = 'stopwords.txt'


class Vector:
    """
    Contains vectorized representations of document
    """
    def __init__(self, query: Optional[bool] = None,
                 words: Optional[pd.Series] = None,
                 loc: Optional[str] = None,
                 corpus: Optional[str] = None,
                 f: Optional[pd.Series] = None,
                 tf: Optional[pd.Series] = None,
                 df: Optional[pd.Series] = None,
                 idf: Optional[pd.Series] = None,
                 n: Optional[int] = None,
                 dl: Optional[int] = None,
                 avdl: Optional[int] = None,
                 tf_idf: Optional[pd.Series] = None):
        self.query = query
        self.words = words
        self.loc = loc
        self.corpus = corpus
        self.f = f
        self.tf = tf
        self.df = df
        self.idf = idf
        self.n = n
        self.dl = dl
        self.avdl = avdl
        self.tf_idf = tf_idf

    def __repr__(self):
        return f"Vector({self.loc}, corpus={self.corpus}, query={self.query})"

    def __str__(self):
        return f"Vector({self.author()}/{self.name()})"

    def cos_sim(self, other):
        """
        Calculates cosine similarity between two Vector TF-IDFs.

        :param other: Another Vector object
        :return: Float representing cosine similarity
        """
        a = self.tf_idf
        b = other.tf_idf

        # t = list(set(self.f[self.f != 0].index)
        #          .intersection(set(other.f[other.f != 0].index)))
        # a = a.loc[t]
        # b = b.loc[t]

        return (a * b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()))

    def okapi(self, other, k1: float = 1.5, k2: float = 1.5):
        """
        Calculates okapi similarity between two Vectors.

        :param other: Another Vector object
        :param k1: (Default 1) Float normalization parameter,
            must be in range [1, 2]
        :param k2: (Default 1) Float normalization parameter,
            must be in range [1, 1000]
        :return:
        """
        if k1 < 1 or k1 > 2:
            raise ValueError("k1 must be in range [1.0, 2.0]")
        if k2 < 1 or k2 > 1000:
            raise ValueError("k2 must be in range [1, 1000]")

        b = 0.75
        t = list(set(self.f[self.f != 0].index)
                 .intersection(set(other.f[other.f != 0].index)))

        part1 = np.log((self.n - self.df.loc[t] + 0.5) / (self.df.loc[t] + 0.5))
        part2 = (((k1 + 1) * self.f.loc[t])
                 / (k1 * (1 - b + b * (self.dl / self.avdl)) + self.f.loc[t]))
        part3 = ((k2 + 1) * other.f.loc[t]) / (k2 + other.f.loc[t])

        return (part1 * part2 * part3).sum()

    def author(self) -> str:
        """
        Gets author name from Vector document filepath.

        :return: String author name
        """
        return self.loc.split("/")[-2]

    def name(self) -> str:
        """
        Gets document name from Vector document filepath

        :return: String document name
        """
        return self.loc.split("/")[-1]


def get_filenames(homedir, size: Optional[int] = None) -> pd.Series:
    """
    Gets all files in a directory, even if they are in a subfolder.

    :param homedir: String representing home directory path
    :param size: (Default None) Optional size of sample from full corpus
    :return: A pandas series of document paths
    """
    docs = []
    for root, dirs, files in os.walk(homedir):
        for file in files:
            docs.append(os.path.join(root, file).replace("\\", "/"))
    docs = pd.Series(docs, name=homedir)
    if size is not None:
        docs = docs.sample(size, ignore_index=True)
    return docs


def get_stopwords(link: str) -> pd.Series:
    """
    Gets stopwords from TXT link online.

    :param link: String representing link of stopwards file
    :return: A pandas series of stopwords
    """
    return pd.read_csv(link, header=None, sep=" ")[0]


def get_stemwords(docs: pd.Series,
                  stopwords: pd.Series = get_stopwords(default_stopwords_link)
                  ) -> pd.Series:
    """
    Uses PorterStemmer to get stemmed words from raw text bodies.

    :param docs: A pandas series of document names
    :param stopwords: A pandas series of stopwords
    :return: A pandas series of stemmed words
    """
    all_lines = []
    for doc in docs:
        with open(doc, 'r', encoding="latin-1") as doctxt:
            lines = [line.strip('\n') for line in doctxt.readlines()]
            all_lines.extend(lines)
    words_step1 = pd.Series(pd.Series(all_lines)
                            .str.lower()
                            .str.replace(r"[^a-zA-Z'.]|^\.|\.$", ' ', regex=True)
                            .str.split()
                            .explode()
                            .str.replace(r"^'|'$|'s$|\.", '', regex=True)
                            .str.strip(" ")
                            .unique()
                            ).sort_values()
    words_step2 = words_step1[(~words_step1.isin(stopwords)) & (words_step1.str.len() > 0)]
    p = PorterStemmer()
    words_step3 = words_step2.apply(lambda w: p.stem(w)).unique()
    return pd.Series(words_step3)


def read_doc(s: str) -> str:
    """
    Reads document from string

    :param s: String filepath
    :return: String document body text
    """
    with open(s, 'r', encoding="latin-1") as doctext:
        return doctext.read()


def stem_doc(s: str):
    """
    Returns stemmed version of document text

    :param s: String filepath
    :return: String document stemmed body text
    """
    p = PorterStemmer()
    return " ".join(pd.Series([s])
                    .str.lower()
                    .str.replace(r"[^a-zA-Z'.]|^\.|\.$", ' ', regex=True)
                    .str.split()
                    .explode()
                    .str.replace(r"^'|'$|'s$|\.", '', regex=True)
                    .str.strip(" ")
                    .apply(lambda w: p.stem(w)))


def ground_truth(docs: pd.Series):
    """
    Creates ground truth dataframe

    :param docs: A pandas series of document filepaths
    :return: A pandas dataframe with document and observed author
    """
    gt = pd.DataFrame()
    docs_split = docs.str.split("/")
    gt['doc'] = docs_split.apply(lambda d: d[-1])
    gt['author'] = docs_split.apply(lambda d: d[-2])

    return gt


def term_freq(docs: pd.Series, words: pd.Series) -> pd.DataFrame:
    """
    Returns term frequency representation of documents

    :param docs: A pandas series of document filepaths
    :param words: A pandas series of words of corpus
    :return: A pandas dataframe where each column is a word and each row is a
        term frequency document representation
    """
    doctexts = docs.apply(read_doc).apply(stem_doc)

    n_words = len(words)
    freqs = []
    for i, word in enumerate(words):
        print(f"Vectorizing {word}... ({i+1}/{n_words})")
        freqs.append(doctexts.str.count(word))
    f = pd.concat(freqs, axis=1)
    f.columns = words

    return f.set_index(docs)


def doc_freq(docs: pd.Series, words: pd.Series,
             f: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Returns document frequency representation of corpus

    :param docs: A pandas series of document filepaths
    :param words: A pandas series of words of corpus
    :param f: (Default None) An optional term frequency pandas dataframe
    :return: A pandas series where each entry is a word's document frequency
    """
    if f is None:
        f = term_freq(docs, words)
    df = f.apply(lambda x: 1 * (x > 0)).sum(axis=0)
    return df


def vector_from_f(docname: str, corpus: str, f_table: pd.DataFrame,
                  words: pd.Series, n: int, df: pd.Series, idf: pd.Series,
                  avdl: int, query: bool = False) -> Vector:
    """
    Generates a Vector object from an appropriately formatted term frequency
    table and corpus-level metrics

    :param docname: Name of the document
    :param corpus: Name of the corpus
    :param f_table: A pandas dataframe representing a term frequency table
    :param words: A pandas series representing a list of words in the term
        frequency table
    :param n: The number of documents in the corpus
    :param df: A pandas series representing document frequency
    :param idf: A pandas series representing inverse document frequency
    :param avdl: The average size of a file in bytes
    :param query: (Default False) Whether or not to use the query formula
        when calculating TF-IDF.
    :return: A Vector instance.
    """
    loc = docname
    f = f_table.loc[docname]
    tf = f.divide(f.max())

    if query:
        tfidf = (0.5 + 0.5 * tf) * idf
    else:
        tfidf = tf * idf

    dl = os.path.getsize(loc)

    return Vector(loc=loc, corpus=corpus, words=words, f=f, tf=tf, df=df,
                  idf=idf, tf_idf=tfidf, n=n, dl=dl, avdl=avdl, query=query)


def vectors_from_f(f_table: pd.DataFrame, query: bool = False) -> pd.Series:
    """
    Generates a pandas series of multiple vectors from a term frequency table

    :param f_table: A pandas series that is a term frequency table. The index
        should be the document names, the index name should be the root
        directory, and the columns are words.
    :param query: (Default False) Whether or not to use the query formula
        when calculating TF-IDF.
    :return: A pandas series of Vector instances.
    """
    words = pd.Series(f_table.columns)
    corpus = f_table.index.name
    n = len(f_table.index)
    df = doc_freq(pd.Series(f_table.index), words, f=f_table)
    idf = np.log2(n / df)

    avdl = pd.Series(f_table.index).apply(os.path.getsize).mean()

    return pd.Series(f_table.index).apply(
        lambda d: vector_from_f(d, corpus, f_table,
                                words, n, df, idf, avdl, query=query))


def main():
    args = sys.argv
    if len(args) < 2:
        print("Syntax: python3 textVectorizer.py <Directory> [<SampleSize>]")
    else:
        homedir = args[1]
        samp_size = None
        if len(args) == 3:
            samp_size = int(args[2])
        docs = get_filenames(homedir, samp_size)
        stopwords = get_stopwords(default_stopwords_link)
        words = get_stemwords(docs, stopwords)
        f = term_freq(docs, words)
        f.to_csv(f"{homedir}_vectorized.csv")

        gt = ground_truth(docs)
        gt.to_csv(f"{homedir}_ground_truth.csv", header=False, index=False)
        print(f"Authors: {len(gt['author'].unique())}")
        print(f"Documents: {len(gt['doc'])}")


if __name__ == "__main__":
    main()
