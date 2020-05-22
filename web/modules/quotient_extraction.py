# basics
import warnings
import operator

# performant computations
from numba import njit

# scientific computations
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix, lil_matrix, find

# data
import numpy as np

# machine learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# custom modules
from modules.spelling_correction import edit_distance


@njit
def pairwise_damerau_levenshtein_distances(corpus, dtype=np.int64):
    n = len(corpus)
    distances = np.empty((n - 1)*n // 2, dtype=dtype)

    k = 0
    for i in range(n):
        for j in range(i):
            distances[k] = edit_distance(corpus[i], corpus[j], transpositions=True)
            k += 1
    
    return distances

def pairwise_damerau_levenshtein_similarities(corpus, distances=None, dtype=np.float64):
    n = len(corpus)
    maximum_lengths = np.empty((n - 1)*n // 2, dtype=np.int64)

    k = 0
    for i in range(n):
        for j in range(i):
            maximum_lengths[k] = max(len(corpus[i]), len(corpus[j]))
            k += 1
    
    if distances is None:
        distances = pairwise_damerau_levenshtein_distances(corpus)
    
    similarities = 1 - distances/maximum_lengths.astype(dtype)
    
    return similarities

def symmetric_matrix(entries, diag=0, dtype=np.float64):
    n = int(1/2 + (1/4 + 2*len(entries))**(1/2))
    symmetric_matrix = np.empty((n, n), dtype=dtype)
    
    for i in range(n):
        symmetric_matrix[i, i] = diag
    k = 0
    for i in range(n):
        for j in range(i):
            symmetric_matrix[i, j] = entries[k]
            k += 1
    
    upper_triangular_indices = np.triu_indices(n, 1)
    symmetric_matrix[upper_triangular_indices] = symmetric_matrix.transpose()[upper_triangular_indices]

    return symmetric_matrix

def csgraph(matrix, threshold=0.):
    return csr_matrix(matrix >= threshold)

def quotient_matrix(csgraph):
    n = csgraph.shape[0]
    
    components = connected_components(csgraph, directed=False)
    projection = dict(zip(range(n), components[1]))
    
    p = lil_matrix((n, components[0]))
    for (i, j) in projection.items():
        p[i, j] = 1
    
    return csr_matrix(p, dtype=np.int64)


class QuotientCountVectorizer(CountVectorizer):
    
    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None, stop_words=None,
                 token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False, dtype=np.int64, distances=None, threshold=0.8,
                 quotient_matrix=None):
        super().__init__(input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents,
                         lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer,
                         stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range,
                         analyzer=analyzer, max_df=max_df, min_df=min_df, max_features=max_features,
                         vocabulary=vocabulary, binary=binary, dtype=dtype)
        self.distances = distances
        self.threshold = threshold
        self.quotient_matrix = quotient_matrix
            
        # to be set in fit method
        self.quotient_matrix_ = None

    def fit(self, corpus, y=None):
        super().fit_transform(corpus) # due to unfortunate implementation of CountVectorizer.fit
        
        if self.quotient_matrix is None:
            vocabulary = np.array(super().get_feature_names()).astype('U')
            similarities = pairwise_damerau_levenshtein_similarities(vocabulary, self.distances)
            strong_similarities = csgraph(symmetric_matrix(similarities), self.threshold)
            self.quotient_matrix_ = quotient_matrix(strong_similarities)
        else:
            self.quotient_matrix_ = self.quotient_matrix
        
        return self
    
    def transform(self, corpus):
        return super().transform(corpus).dot(self.quotient_matrix_)
    
    def fit_transform(self, corpus, y=None):
        X = super().fit_transform(corpus)

        if self.quotient_matrix is None:
            vocabulary = np.array(super().get_feature_names()).astype('U')
            similarities = pairwise_damerau_levenshtein_similarities(vocabulary, self.distances)
            strong_similarities = csgraph(symmetric_matrix(similarities), self.threshold)
            self.quotient_matrix_ = quotient_matrix(strong_similarities)
        else:
            self.quotient_matrix_ = self.quotient_matrix

        return X.dot(self.quotient_matrix_)
    
    
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py
class QuotientTfidfVectorizer(QuotientCountVectorizer):

    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, 
                 preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1),
                 analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.int64,
                 norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False, distances=None, threshold=0.8,
                 quotient_matrix=None):

        super().__init__(input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents,
                         lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
                         stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df,
                         min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype,
                         distances=distances, threshold=threshold, quotient_matrix=quotient_matrix)

        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value

    @property
    def idf_(self):
        return self._tfidf.idf_

    @idf_.setter
    def idf_(self, value):
        self._validate_vocabulary()
        if hasattr(self, 'vocabulary_'):
            if len(self.vocabulary_) != len(value):
                raise ValueError("idf length = %d must be equal "
                                 "to vocabulary size = %d" %
                                 (len(value), len(self.vocabulary)))
        self._tfidf.idf_ = value

    def fit(self, corpus, y=None):
        X = super().fit_transform(corpus)
        self._tfidf.fit(X)
        return self

    def transform(self, corpus):
        X = super().transform(corpus)
        return self._tfidf.transform(X, copy=False)
    
    def fit_transform(self, corpus, y=None):
        X = super().fit_transform(corpus)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)
