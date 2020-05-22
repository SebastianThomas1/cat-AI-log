# Sebastian Thomas (datascience at sebastianthomas dot de)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler


class Vectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, type='count'):
        if type == 'count':
            self._vectorizer = CountVectorizer()
        elif type == 'hashing':
            self._vectorizer = HashingVectorizer(alternate_sign=False)
        elif type == 'tfidf':
            self._vectorizer = TfidfVectorizer()
    
    def fit(self, corpus, y=None):
        self._vectorizer.fit(corpus)
        return self
    
    def transform(self, corpus, y=None):
        return self._vectorizer.transform(corpus)
    
class Scaler(BaseEstimator, TransformerMixin):

    def __init__(self, type='standard'):
        if type == 'standard':
            self._scaler = StandardScaler(with_mean=False)
        elif type == 'maxabs':
            self._scaler = MaxAbsScaler()
        elif type == 'minmax':
            self._scaler = MinMaxScaler()
        elif type == 'norm':
            self._scaler = Normalizer()
        elif type == 'robust':
            self._scaler = RobustScaler(with_centering=False)
        elif type == None:
            self._scaler = None
   
    def fit(self, X, y=None):
        if self._scaler is not None:
            self._scaler.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self._scaler.transform(X) if self._scaler is not None else X