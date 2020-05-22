# Sebastian Thomas (datascience at sebastianthomas dot de)

# scientific computations
from scipy.sparse import find

# data
import numpy as np

# machine learning
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer, Normalizer

# custom modules
from modules.quotient_extraction import pairwise_damerau_levenshtein_similarities, csgraph, symmetric_matrix, quotient_matrix


class SearchEngine:
    
# to do: DocString!

    def __init__(self, lowercase=True, token_pattern=r'(?u)\b\w+\b', vocabulary=None, correct_spelling=True, distances=None,
                 threshold=0.8, quotient_matrix=None):
        self._count_vectorizer = CountVectorizer(lowercase=lowercase, token_pattern=token_pattern, vocabulary=vocabulary)
        self._normalizer = Normalizer()
        self._normal_strict_vectorizer = make_pipeline(self._count_vectorizer, self._normalizer)
        
        self.correct_spelling = correct_spelling

        self.distances = distances
        self.threshold = threshold
        self.quotient_matrix = quotient_matrix
            
        # to be set in fit method
        self.corpus_ = None
        self.quotient_matrix_ = None
        self.nsdt_ = None
        self.nqdt_ = None
        self._normal_quotient_vectorizer = None
    
    def _quotient_morphism(self, instance):
        return instance.dot(self.quotient_matrix_)
    
    def fit(self, corpus):
        self.corpus_ = np.array(corpus).astype(str)
        
        # fit count vectorizer
        X = self._count_vectorizer.fit_transform(self.corpus_)
        
        # compute strict document term matrix
        self.nsdt_ = self._normalizer.transform(X)
        
        if self.correct_spelling:
            # determine quotient matrix
            if self.quotient_matrix is None:
                vocabulary = np.array(self._count_vectorizer.get_feature_names()).astype('U')
                similarities = pairwise_damerau_levenshtein_similarities(vocabulary, self.distances)
                strong_similarities = csgraph(symmetric_matrix(similarities), self.threshold)
                self.quotient_matrix_ = quotient_matrix(strong_similarities)
            else:
                self.quotient_matrix_ = self.quotient_matrix

            # define quotient pipeline
            self._quotient_transformer = FunctionTransformer(self._quotient_morphism)
            self._normal_quotient_vectorizer = make_pipeline(self._count_vectorizer, self._quotient_transformer, self._normalizer)
        
            # compute quotient document term matrix
            self.nqdt_ = self._normalizer.transform(self._quotient_transformer.transform(X))

        return self

    def recommend(self, query, max_count=1, threshold=0., output='documents', correct_spelling=True, include_all=False):
        if max_count is None:
            max_count = self.corpus_.size
            
        if self.correct_spelling and correct_spelling:
            query_vector = self._normal_quotient_vectorizer.transform([query])
            ndt = self.nqdt_
        else:
            query_vector = self._normal_strict_vectorizer.transform([query])
            ndt = self.nsdt_
        
        # compute cosine similarities
        similarities = ndt.dot(query_vector.transpose())
        
        # restrict to those instances that contain all query tokens
        if include_all:
            similarities = similarities.multiply(np.sum(ndt.multiply(query_vector) > 0, axis=1) == np.sum(query_vector > 0))
        
        # extract indices and positive cosine similarities from sparse array
        (indices, _, positive_similarities) = find(similarities)
                
        # restrict to those instances resp. similarities over the threshold
        indices = indices[positive_similarities >= threshold]
        positive_similarities = positive_similarities[positive_similarities >= threshold]
        
        recommended_indices = indices[positive_similarities.argsort()[::-1]]
        if output == 'indices':
            return recommended_indices[:max_count]
        else:
            recommended_documents = self.corpus_[recommended_indices]
            if output == 'documents':
                return recommended_documents[:max_count]
            elif output == 'with_similarities':
                ranked_positive_similarities = positive_similarities[positive_similarities.argsort()[::-1]]
                return np.concatenate([recommended_documents[:max_count].reshape(-1, 1),
                                       ranked_positive_similarities[:max_count].reshape(-1, 1)],
                                      axis=1)