# Sebastian Thomas (datascience at sebastianthomas dot de)

# scientific computations
#import scipy as sp

# data
#import numpy as np

# custom modules
from modules.spelling_correction import *

# unit tests
from unittests.general_tests import *


def test_damerau_levenshtein_distance(debug=False):
    successes = 0
    failures = 0
    
    for test in [test_equality(damerau_levenshtein_distance('Hello', 'HELLO'), 4, debug=debug),
                 test_equality(damerau_levenshtein_distance('Hello', 'Hell'), 1, debug=debug),
                 test_equality(damerau_levenshtein_distance('Hello', 'Salut'), 4, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
            
    return successes, failures


def test_spelling_corrector(debug=False):
    successes = 0
    failures = 0
    
    corpus = ['Hello World', 'Salut']
    spelling_corrector = SpellingCorrector()
    spelling_corrector.fit(corpus)
    
    for test in [test_numpy_array_entries(spelling_corrector.transform(['Hello World', 'HELLO World',
                                                                        'Hell World']),
                                          ['Hello World']*3, strict=True, debug=debug),
                 test_numpy_array_entries(spelling_corrector.transform(['Salut', 'Salt']),
                                          ['Salut']*2, strict=True, debug=debug)]:
        if test:
            successes += 1
        else:
            failures += 1
            
    return successes, failures


tests = [test_damerau_levenshtein_distance,
         test_spelling_corrector]