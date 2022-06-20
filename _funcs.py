"""Support functions to both _ufuncs and _afuncs"""

from .exceptions import IncompatibleDimensions

def output_pdim(pdim):
    return None if pdim.is_dimensionless else pdim

# Convenience functions for testing for same dimensionality 

def same_pdim(a,b):
    try:
        return a.pdim == b.pdim
    except:
        return False

def assert_same_pdim(a,b):
    if not same_pdim(a,b):
        raise IncompatibleDimensions(a,b)

