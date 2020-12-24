# Sebastian Thomas (datascience at sebastianthomas dot de)

# scientific computations
import scipy as sp

# data
import numpy as np

# custom modules
from modules.quotient_extraction import *

# unit tests
from unittests.general_tests import *


def test_pairwise_damerau_levenshtein_distances(debug=False):
    successes = 0
    failures = 0
    
    vocabulary = ['Hello', 'HELLO', 'Helo', 'Salut']
    distances = pairwise_damerau_levenshtein_distances(vocabulary)
    for test in [test_type(distances, np.ndarray, debug=debug),
                 test_length(distances, 6, debug=debug),
                 test_numpy_array_entries(distances, np.array([4, 1, 4, 4, 5, 4]), strict=True, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1

    vocabulary = [word.lower() for word in vocabulary]
    distances = pairwise_damerau_levenshtein_distances(vocabulary)
    for test in [test_type(distances, np.ndarray, debug=debug),
                 test_length(distances, 6, debug=debug),
                 test_numpy_array_entries(distances, np.array([0, 1, 1, 4, 4, 4]), strict=True, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
    
    return successes, failures


def test_pairwise_damerau_levenshtein_similarities(debug=False):
    successes = 0
    failures = 0
    
    vocabulary = ['Hello', 'HELLO', 'Helo', 'Salut']
    similarities = pairwise_damerau_levenshtein_similarities(vocabulary)
    for test in [test_type(similarities, np.ndarray, debug=debug),
                 test_length(similarities, 6, debug=debug),
                 test_numpy_array_entries(similarities, np.array([0.2, 0.8, 0.2, 0.2, 0., 0.2]), debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1

    vocabulary = [word.lower() for word in vocabulary]
    similarities = pairwise_damerau_levenshtein_similarities(vocabulary)
    for test in [test_type(similarities, np.ndarray, debug=debug),
                 test_length(similarities, 6, debug=debug),
                 test_numpy_array_entries(similarities, np.array([1., 0.8, 0.8, 0.2, 0.2, 0.2]), debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
        
    return successes, failures


def test_symmetric_matrix(debug=False):
    successes = 0
    failures = 0
    
    matrix = symmetric_matrix(np.array([4, 1, 4, 4, 5, 4]), dtype=np.int64)
    for test in [test_type(matrix, np.ndarray, debug=debug),
                 test_shape(matrix, (4, 4), debug=debug),
                 test_numpy_array_entries(matrix, np.array([[0, 4, 1, 4],
                                                            [4, 0, 4, 5],
                                                            [1, 4, 0, 4],
                                                            [4, 5, 4, 0]]), debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1

    matrix = symmetric_matrix(np.array([1., 0.8, 0.8, 0.2, 0.2, 0.2]), diag=1.)
    for test in [test_type(matrix, np.ndarray, debug=debug),
                 test_shape(matrix, (4, 4), debug=debug),
                 test_numpy_array_entries(matrix, np.array([[1. , 1. , 0.8, 0.2],
                                                            [1. , 1. , 0.8, 0.2],
                                                            [0.8, 0.8, 1. , 0.2],
                                                            [0.2, 0.2, 0.2, 1. ]]), debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
        
    return successes, failures


def test_csgraph(debug=False):
    successes = 0
    failures = 0
    
    matrix = np.array([[1. , 1. , 0.8, 0.2],
                       [1. , 1. , 0.8, 0.2],
                       [0.8, 0.8, 1. , 0.2],
                       [0.2, 0.2, 0.2, 1. ]])
    graph = csgraph(matrix, threshold=0.8)
    for test in [test_type(graph, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(graph, (4, 4), debug=debug),
                 test_stored_values(graph, 10, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
       
    matrix = np.array([[1. , 0.2, 0.8, 0.2],
                       [0.2, 1. , 0.2, 0. ],
                       [0.8, 0.2, 1. , 0.2],
                       [0.2, 0. , 0.2, 1. ]])
    graph = csgraph(matrix, threshold=0.8)
    for test in [test_type(graph, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(graph, (4, 4), debug=debug),
                 test_stored_values(graph, 6, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
       
    return successes, failures


def test_quotient_matrix(debug=False):
    successes = 0
    failures = 0
    
    matrix = np.array([[1. , 1. , 0.8, 0.2],
                       [1. , 1. , 0.8, 0.2],
                       [0.8, 0.8, 1. , 0.2],
                       [0.2, 0.2, 0.2, 1. ]])
    graph = csgraph(matrix, threshold=0.8)
    q_matrix = quotient_matrix(graph)
    for test in [test_type(q_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(q_matrix, (4, 2), debug=debug),
                 test_stored_values(q_matrix, 4, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
       
    matrix = np.array([[1. , 0.2, 0.8, 0.2],
                       [0.2, 1. , 0.2, 0. ],
                       [0.8, 0.2, 1. , 0.2],
                       [0.2, 0. , 0.2, 1. ]])
    graph = csgraph(matrix, threshold=0.8)
    q_matrix = quotient_matrix(graph)
    for test in [test_type(q_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(q_matrix, (4, 3), debug=debug),
                 test_stored_values(q_matrix, 4, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
       
    return successes, failures


def test_quotient_count_vectorizer(debug=False):
    successes = 0
    failures = 0
    
    corpus = ['Hello World', 'HELLO WORLD', 'Hell World', 'Salut']
    quotient_count_vectorizer = QuotientCountVectorizer(lowercase=False)
    quotient_count_vectorizer.fit(corpus)
    q_matrix = quotient_count_vectorizer.quotient_matrix_
    document_term_matrix = quotient_count_vectorizer.transform(corpus)
    for test in [test_type(q_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(q_matrix, (6, 5), debug=debug),
                 test_stored_values(q_matrix, 6, debug=debug),
                 test_type(document_term_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(document_term_matrix, (4, 5), debug=debug),
                 test_stored_values(document_term_matrix, 7, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
       
    quotient_count_vectorizer = QuotientCountVectorizer()
    quotient_count_vectorizer.fit(corpus)
    q_matrix = quotient_count_vectorizer.quotient_matrix_
    document_term_matrix = quotient_count_vectorizer.transform(corpus)
    for test in [test_type(q_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(q_matrix, (4, 3), debug=debug),
                 test_stored_values(q_matrix, 4, debug=debug),
                 test_type(document_term_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(document_term_matrix, (4, 3), debug=debug),
                 test_stored_values(document_term_matrix, 7, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
       
    return successes, failures


def test_quotient_tfidf_vectorizer(debug=False):
    successes = 0
    failures = 0
    
    corpus = ['Hello World', 'HELLO WORLD', 'Hell World', 'Salut']
    quotient_tfidf_vectorizer = QuotientTfIdfVectorizer(lowercase=False)
    quotient_tfidf_vectorizer.fit(corpus)
    q_matrix = quotient_tfidf_vectorizer.quotient_matrix_
    document_term_matrix = quotient_tfidf_vectorizer.transform(corpus)
    for test in [test_type(q_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(q_matrix, (6, 5), debug=debug),
                 test_stored_values(q_matrix, 6, debug=debug),
                 test_type(document_term_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(document_term_matrix, (4, 5), debug=debug),
                 test_stored_values(document_term_matrix, 7, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
       
    quotient_tfidf_vectorizer = QuotientTfIdfVectorizer()
    quotient_tfidf_vectorizer.fit(corpus)
    q_matrix = quotient_tfidf_vectorizer.quotient_matrix_
    document_term_matrix = quotient_tfidf_vectorizer.transform(corpus)
    for test in [test_type(q_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(q_matrix, (4, 3), debug=debug),
                 test_stored_values(q_matrix, 4, debug=debug),
                 test_type(document_term_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(document_term_matrix, (4, 3), debug=debug),
                 test_stored_values(document_term_matrix, 7, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
       
    return successes, failures


tests = [test_pairwise_damerau_levenshtein_distances,
         test_pairwise_damerau_levenshtein_similarities,
         test_symmetric_matrix,
         test_csgraph,
         test_quotient_matrix,
         test_quotient_count_vectorizer,
         test_quotient_count_vectorizer]
