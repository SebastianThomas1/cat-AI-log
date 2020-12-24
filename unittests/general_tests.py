# Sebastian Thomas (datascience at sebastianthomas dot de)

#data
import numpy as np


RED = '\033[91m'


def test_equality(x, y, debug=False):
    success = x == y

    if not success and debug:
        print(RED + '{} != {}'.format(x, y))
    
    return success


def test_type(x, t, debug=False):
    success = isinstance(x, t)
    
    if not success and debug:
        print(RED + '{} is not of type {}'.format(x, t))
    
    return success


def test_length(x, length, debug=False):
    success = len(x) == length
    
    if not success and debug:
        print(RED + '{} is not of length {}'.format(x, length))
    
    return success


def test_shape(x, shape, debug=False):
    success = x.shape == shape
    
    if not success and debug:
        print(RED + '{} is not of shape {}'.format(x, shape))
    
    return success


def test_numpy_array_entries(x, array, strict=False, rtol=1e-05, atol=1e-08, equal_nan=False, debug=False):
    success = np.all(x == array) if strict else np.allclose(x, array, rtol=rtol, atol=atol, equal_nan=equal_nan)
    
    if not success and debug:
        print(RED + '{} has not the same entries as {}'.format(x, array))
        
    return success


def test_stored_values(x, count, debug=False):
    success = x.getnnz() == count
    
    if not success and debug:
        print(RED + '{} has not {} stored values'.format(x, count))
        
    return success