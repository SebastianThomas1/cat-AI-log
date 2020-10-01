# Sebastian Thomas (datascience at sebastianthomas dot de)

# scientific computations
import scipy as sp

# data
import numpy as np

# custom modules
from modules.search import *
from modules.general_tests import *


def test_search_engine(debug=False):
    successes = 0
    failures = 0
    
    corpus = ['Hello World', 'HELLO WORLD', 'Hell World', 'Salut']
    search_engine = SearchEngine(lowercase=False)
    search_engine.fit(corpus)
    q_matrix = search_engine.quotient_matrix_
    s_document_term_matrix = search_engine.nsdt_
    q_document_term_matrix = search_engine.nqdt_
    for test in [test_type(q_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(q_matrix, (6, 5), debug=debug),
                 test_stored_values(q_matrix, 6, debug=debug),
                 test_type(s_document_term_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(s_document_term_matrix, (4, 6), debug=debug),
                 test_stored_values(s_document_term_matrix, 7, debug=debug),
                 test_type(q_document_term_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(q_document_term_matrix, (4, 5), debug=debug),
                 test_stored_values(q_document_term_matrix, 7, debug=debug),
                 test_numpy_array_entries(search_engine.recommend('Hello'), 
                                          np.array(['Hell World']),
                                          strict=True, debug=debug),
                 test_numpy_array_entries(search_engine.recommend('Hello', max_count=None), 
                                          np.array(['Hell World', 'Hello World']),
                                          strict=True, debug=debug),
                 test_numpy_array_entries(search_engine.recommend('Hello', max_count=None, output='indices'), 
                                          np.array([2, 0]), strict=True, debug=debug),
                 test_numpy_array_entries(search_engine.recommend('Hello', max_count=None,
                                                                  output='with_similarities'), 
                                          np.array([['Hell World', '0.7071067811865475'],
                                                    ['Hello World', '0.7071067811865475']]),
                                          strict=True, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1

    corpus = ['Hello World', 'HELLO WORLD', 'Hell World', 'Salut']
    search_engine = SearchEngine()
    search_engine.fit(corpus)
    q_matrix = search_engine.quotient_matrix_
    s_document_term_matrix = search_engine.nsdt_
    q_document_term_matrix = search_engine.nqdt_
    for test in [test_type(q_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(q_matrix, (4, 3), debug=debug),
                 test_stored_values(q_matrix, 4, debug=debug),
                 test_type(s_document_term_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(s_document_term_matrix, (4, 4), debug=debug),
                 test_stored_values(s_document_term_matrix, 7, debug=debug),
                 test_type(q_document_term_matrix, sp.sparse.csr.csr_matrix, debug=debug),
                 test_shape(q_document_term_matrix, (4, 3), debug=debug),
                 test_stored_values(q_document_term_matrix, 7, debug=debug),
                 test_numpy_array_entries(search_engine.recommend('Hello'), 
                                          np.array(['Hell World']),
                                          strict=True, debug=debug),
                 test_numpy_array_entries(search_engine.recommend('Hello', max_count=None), 
                                          np.array(['Hell World', 'HELLO WORLD', 'Hello World']),
                                          strict=True, debug=debug),
                 test_numpy_array_entries(search_engine.recommend('Hello', max_count=None, output='indices'), 
                                          np.array([2, 1, 0]), strict=True, debug=debug),
                 test_numpy_array_entries(search_engine.recommend('Hello', max_count=None,
                                                                  output='with_similarities'), 
                                          np.array([['Hell World', '0.7071067811865475'],
                                                    ['HELLO WORLD', '0.7071067811865475'],
                                                    ['Hello World', '0.7071067811865475']]),
                                          strict=True, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
            
    return (failures == 0, successes, failures)


def print_test_module_search(debug=False):
    results = [test_search_engine(debug=debug)]
    names = ['search_engine']

    for result, name in zip(results, names):
        all_passed, successes, failures = result
        print('\033[90mTesting {}:'.format(name))
        if failures == 0:
            print('\033[92mAll tests passed')
        else:
            print('\033[92m{} tests passed'.format(successes))
            print('\033[91m{} tests failed'.format(failures))
