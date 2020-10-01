# Sebastian Thomas (datascience at sebastianthomas dot de)

# basics
import warnings
import operator

# performant computations
from numba import njit

# regular expressions
import re

# data
import numpy as np

# machine learning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


# from https://www.nltk.org/_modules/nltk/metrics/distance.html
@njit
def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i  # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j  # row 0: 0,1,2,3,4,...
    return lev


# from https://www.nltk.org/_modules/nltk/metrics/distance.html
@njit
def _edit_dist_step(lev, i, j, s1, s2, substitution_cost=1, transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    # substitution
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    lev[i][j] = min(a, b, c, d)

    
# from https://www.nltk.org/_modules/nltk/metrics/distance.html
@njit
def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(lev, i + 1, j + 1, s1, s2, substitution_cost=substitution_cost,
                            transpositions=transpositions)
    return lev[len1][len2]


@njit
def damerau_levenshtein_distance(s1, s2):
    return edit_distance(s1, s2, substitution_cost=1, transpositions=True)


class SpellingCorrector(BaseEstimator, TransformerMixin):
    
# to do: DocString!

    def __init__(self, word_distance='damlev', lowercase=False, token_pattern=r'(?u)\b\w+\b'):
        if word_distance == 'damlev':
            self.word_distance = damerau_levenshtein_distance
        elif word_distance == 'lev':
            self.word_distance = edit_distance
            
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        
        # to be set in fit method
        self.vocabulary_ = None
        self.low_vocabulary_ = None

    def _nearest_token(self, word):
        if self.lowercase:
            low_word = word.lower()
            # only determine the nearest token if the word is not in the vocabulary
            return (word if word in self.vocabulary_
                    else self.vocabulary_[np.argmin([self.word_distance(low_word, low_token)
                                                     for low_token in self.low_vocabulary_])])
        else:
            return (word if word in self.vocabulary_
                    else self.vocabulary_[np.argmin([self.word_distance(word, token)
                                                     for token in self.vocabulary_])])
    
    def _tokenize(self, document):
        return re.findall(self.token_pattern, document)
    
    def _transform(self, document):
        return ' '.join([self._nearest_token(word) for word in self._tokenize(document)])
    
    def fit(self, corpus):
        vectorizer = CountVectorizer(lowercase=False, token_pattern=self.token_pattern)
        vectorizer.fit(corpus)
        self.vocabulary_ = np.array(vectorizer.get_feature_names())
        if self.lowercase:
            self.low_vocabulary_ = np.array([token.lower() for token in self.vocabulary_])

        return self
    
    def transform(self, corpus):
        return np.array([self._transform(document) for document in corpus])