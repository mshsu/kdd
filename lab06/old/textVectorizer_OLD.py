import pandas as pd
import numpy as np
import os
import json
import sys

from nltk.stem.porter import PorterStemmer
from typing import Optional


default_stopwords_link = 'stopwords-long.txt'


class Corpus:
    """
    Represents a corpus of documents
    """
    def __init__(self, loc: Optional[str] = None,
                 docs: Optional[pd.Series] = None,
                 words: Optional[pd.Series] = None,
                 stopwords: Optional[pd.Series] = None,
                 n: Optional[int] = None,
                 df: Optional[pd.Series] = None,
                 idf: Optional[pd.Series] = None,
                 avdl: Optional[int] = None):
        self.loc = loc
        self.docs = docs
        self.words = words
        self.stopwords = stopwords
        self.n = n
        self.df = df
        self.idf = idf
        self.avdl = avdl

    def __repr__(self):
        return f"Corpus({self.loc})"

    def fit(self, homedir: str, size: Optional[int] = None,
                 stopwords: str = default_stopwords_link) -> None:
        """
        Fits Corpus to collection of documents

        :param homedir: A string representing the home directory of the corpus
        :param size: (Default None) Optional size of sample from full corpus
        :param stopwords: A pandas series of stopwords
        :return: None
        """
        self.loc = homedir.replace("/", "\\")
        self.docs = get_filenames(homedir, size=size)
        self.stopwords = get_stopwords(stopwords)
        self.words = get_stemwords(self.docs, stopwords=self.stopwords)
        self.n = len(self.docs)
        self.df = doc_freq(self.docs, self.words)
        self.idf = np.log2(self.n / self.df)
        self.avdl = self.docs.apply(os.path.getsize).mean()

    def copy(self):
        """
        Makes copy of Corpus.

        :return: Corpus that is copy of self
        """
        return Corpus(self.loc, self.docs, self.words, self.stopwords,
                      self.n, self.df, self.idf, self.avdl)

    def pull(self, doc: str, query: bool = False, rm: bool = False):
        """
        Pulls the Vector representation of a specified document from the Corpus

        :param doc: Filepath of document
        :param query: (Default False) If true, treats document as a query when
            calculating its TF-IDF representation
        :param rm: (Default False) If true, calculates Vector representations
            as if the document is not in the Corpus. (Good for leave-one-out)
        :return: A Vector representation of the specified document
        """
        query = query
        corpus = self.copy()

        if rm is True and doc in list(self.docs):
            corpus.rm(doc)
        doc_srs = pd.Series([doc])

        words = corpus.words
        loc = doc.replace("\\", "/")
        f = term_freq(doc_srs, words)
        tf = norm_term_freq(doc_srs, words, f=f)

        df = corpus.df
        idf = corpus.idf
        n = corpus.n
        dl = os.path.getsize(doc)
        avdl = corpus.avdl

        tfidf = tf_idf(doc_srs, corpus.words,
                       tf=tf, idf=idf, query=query).iloc[0]
        f = f.iloc[0]
        tf = tf.iloc[0]

        return Vector(query, words, loc, corpus.loc,
                      f, tf, df, idf, n, dl, avdl, tfidf)

    def pull_all(self, query: bool = False, rm: bool = False, json: bool = False):
        """
        Pulls Vector representation of all documents in corpus

        :param query: (Default False) If true, treats document as a query when
            calculating its TF-IDF representation
        :param json: (Default False) If true, pulls documents as their
            Vector dictionary representations
        :param rm: (Default False) If true, calculates Vector representations
            as if the document is not in the Corpus. (Good for leave-one-out)
        :return: A list of vectors or a list of dictionaries
        """
        if not json:
            return self.docs.apply(lambda d: self.pull(d, query=query, rm=rm)
                                   ).tolist()
        else:
            return self.docs.apply(lambda d: self.pull(d, query=query, rm=rm)
                                   .to_dict()).tolist()

    def to_json(self, output: Optional[str] = None,
                query: bool = False, rm: bool = False) -> None:
        """
        Converts Corpus to JSON/dictionary representation of Vector objects
        and outputs to file

        :param output: (Default None) Filepath to output
        :param query: (Default False) If true, treats document as a query when
            calculating its TF-IDF representation
        :param rm: (Default False) If true, calculates Vector representations
            as if the document is not in the Corpus. (Good for leave-one-out)
        :return: None
        """
        corp = self.pull_all(json=True)
        if output is not None:
            with open(output, 'w') as file:
                json.dump(corp, file)

    def rm(self, doc: str) -> None:
        """
        Removes specified document from Corpus and recalculates attributes
        without the specified document.

        :param doc: Filepath of document to remove
        :return: None
        """
        if doc in list(self.docs):
            self.docs = self.docs[~self.docs.isin([doc])]
            self.words = get_stemwords(self.docs, stopwords=self.stopwords)
            self.n = len(self.docs)
            self.df = doc_freq(self.docs, self.words)
            self.idf = np.log2(self.n / self.df)
            self.avdl = self.docs.apply(os.path.getsize).mean()

    def ground_truth(self) -> pd.DataFrame:
        gt = pd.DataFrame()
        splitname = self.docs.str.split("/")
        gt['filename'] = splitname.apply(lambda l: l[-1])
        gt['author'] = splitname.apply(lambda l: l[-2])
        return gt


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

    def to_dict(self) -> dict:
        """
        Turns Vector into dictionary representation.

        :return: Dictionary representation of Vector
        """
        return {'loc': self.loc, 'corpus': self.corpus,
                'words': list(self.words),
                'f': dense_to_sparse(list(self.f)),
                'df': dense_to_sparse(list(self.df)),
                'n': self.n, 'avdl': self.avdl,
                'query': self.query}

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

    def okapi(self, other, k1: float = 1, k2: float = 1):
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
        if k2 <1 or k2 > 1000:
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
    docs = pd.Series(docs)
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
        with open(doc, 'r') as doctxt:
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
    with open(s, 'r') as doctext:
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


def term_freq(docs: pd.Series, words: pd.Series) -> pd.DataFrame:
    """
    Returns term frequency representation of documents

    :param docs: A pandas series of document filepaths
    :param words: A pandas series of words of corpus
    :return: A pandas dataframe where each column is a word and each row is a
        term frequency document representation
    """
    doctexts = docs.apply(read_doc).apply(stem_doc)

    f = pd.DataFrame()
    f['docname'] = docs
    for word in words:
        f[word] = doctexts.str.count(word)
        f = f.copy()

    return f.set_index('docname')


def norm_term_freq(docs, words, f = None):
    """
    Returns normalized term frequency representation of documents

    :param docs: A pandas series of document filepaths
    :param words: A pandas series of words of corpus
    :param f: (Default None) An optional term frequency pandas dataframe
    :return: A pandas dataframe where each column is a word and each row is a
        normalized term frequency document representation
    """
    if f is None:
        f = term_freq(docs, words)
    return f.divide(f.max(axis=1), axis=0)


def doc_freq(docs, words, f = None):
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


def tf_idf(docs: pd.Series, words: pd.Series, n: Optional[int] = None,
           f: Optional[pd.DataFrame] = None, tf: Optional[pd.DataFrame] = None,
           df: Optional[pd.Series] = None, idf: Optional[pd.Series] = None,
           query: bool = False):
    """
    Returns TF-IDF representation of documents

    :param docs: A pandas series of document filepaths
    :param words: A pandas series of words of corpus
    :param n: (Default None) An optional int denoting number of documents in
        corpus
    :param f: (Default None) An optional term frequency pandas dataframe
    :param tf: (Default None) An optional normalized term frequency pandas
        dataframe
    :param df: (Default None) An optional document frequency pandas series
    :param idf: (Default None) An optional inverse document frequency pandas
        series
    :param query: (Default False) If true, treats document as a query when
            calculating its TF-IDF representation
    :return: A pandas dataframe where each column is a word and each row is a
        TF-IDF document representation
    """
    if f is None and tf is None:
        f = term_freq(docs, words)
    if tf is None:
        tf = norm_term_freq(docs, words, f=f)

    if df is None and idf is None:
        df = doc_freq(docs, words, f=f)
    if n is None and idf is None:
        n = len(tf.index)
    if idf is None:
        idf = np.log2(n / df)

    if query:
        tf_idf = (0.5 + 0.5*tf).multiply(idf, axis=1)
    else:
        tf_idf = tf.multiply(idf, axis=1)
    return tf_idf


def vector_from_dict(d):
    """
    Constructs Vector object from its dictionary representation

    :param d: A dictionary
    :return: A Vector object
    """
    loc = d['loc']
    corpus = d['corpus']
    words = pd.Series(d['words'])

    f = pd.Series(sparse_to_dense(d['f']), index=words, name=loc)
    tf = f.divide(f.max())

    n = d['n']
    df = pd.Series(sparse_to_dense(d['df']), index=words, name=loc)
    idf = np.log2(n / df)

    query = d['query']
    if query:
        tfidf = (0.5 + 0.5 * tf) * idf
    else:
        tfidf = tf * idf

    dl = os.path.getsize(loc)
    avdl = d['avdl']

    return Vector(loc=loc, corpus=corpus, words=words, f=f, tf=tf, df=df,
                  idf=idf, tf_idf=tfidf, n=n, dl=dl, avdl=avdl, query=query)


def dense_to_sparse(dense: list):
    """
    Turns list into sparse dictionary representation, where the dict keys are
    unique values and the dict values are the indices in the dense
    representation where the key value is found.

    If 0s are found in the dense representation, only the maximum index of the
    0s is included.

    :param dense: A list of ints
    :return: A dictionary dense representation of the list of ints
    """
    sparse = {value: [i for i, x in enumerate(dense) if x == value] for value in set(dense)}
    if 0 in sparse.keys():
        sparse[0] = [max(sparse[0])]
    return sparse


def sparse_to_dense(sparse: dict):
    """
    Converts sparse representation back into dense.

    :param sparse: A dictionary dense representation of a list of ints
    :return: The dense list of ints
    """
    max_idx = max(index for indices in sparse.values() for index in indices)
    dense = [0] * (max_idx + 1)

    for value, indices in sparse.items():
        for index in indices:
            dense[index] = int(value)

    return dense


def vectors_from_json(filename: str):
    """
    Converts a JSON representation of Vector objects into a list of Vector
    objects.

    :param filename: JSON file of Vectors represented as dictionaries
    :return: List of Vector objects
    """
    with open(filename, 'r') as file:
        vecs_dict = json.load(file)

    return pd.Series(vecs_dict).apply(vector_from_dict).tolist()


def main():
    args = sys.argv
    if len(args) < 2:
        print("Syntax: python3 textVectorizer.py <Directory> [<SampleSize>]")
    else:
        root = args[1]
        samp_size = None
        if len(args) == 3:
            samp_size = int(args[2])
        corp = Corpus()
        corp.fit(root, size=samp_size)
        corp.ground_truth().to_csv("data_ground_truth.csv",
                                   header=False, index=False)
        corp.to_json("data_vectorized.json")


if __name__ == "__main__":
    main()
