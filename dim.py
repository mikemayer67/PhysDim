"""Physical Dimensionality

This class captures the physical dimensionality of a measurable quantity
without scale.  It does so by tracking the relevant exponents on 7 
fundamental physical properties:

    - Length (L)
    - Time (T)
    - Mass (M)
    - Angle (A)
    - Electric charge (E)
    - Absolute temperature (K)
    - Intensity of light (I)

Internally, it stores these exponents as a tuple in the order (L,T,M,A,E,K,I).
"""

import numpy as np
import numbers

from .exceptions import IncompatibleDimensions
from .exceptions import NotDimLike

_DERIVED_DIMS = {
    (1,0,0,0,0,0,0):'length',
    (0,1,0,0,0,0,0):'time',
    (0,0,1,0,0,0,0):'mass',
    (0,0,0,1,0,0,0):'angle',
    (0,0,0,0,1,0,0):'charge',
    (0,0,0,0,0,1,0):'temperature',
    (0,0,0,0,0,0,1):'illumination',
    (2,0,0,0,0,0,0):'area',
    (3,0,0,0,0,0,0):'volume',
    (0,0,0,2,0,0,0):'solid angle',
    (0,-1,0,0,0,0,0):'frequency',
    (-3,0,1,0,0,0,0):'density',
    (1,-1,0,0,0,0,0):'velocity',
    (0,-1,0,1,0,0,0):'angular velocity',
    (1,-2,0,0,0,0,0):'acceleration',
    (1,-1,1,0,0,0,0):'momentum',
    (1,-2,1,0,0,0,0):'force',
    (2,-2,1,0,0,0,0):'energy',
    (2,-3,1,0,0,0,0):'power',
    (-1,-2,1,0,0,0,0):'pressure',
}


class Dim(object):
    __slot__ = ('_exp')

    def __init__(self, dim=None, *, 
                 length=0, time=0, mass=0, angle=0, charge=0, temp=0, illum=0 ):
        """Dim constructor

        A Dim object can be contructed by one of three methods:

            - specify the exponents for each of the fundamental properties:
              - length (number): optional 
              - time (number): optional 
              - mass (number): optional 
              - angle (number): optional 
              - charge (number): optional 
              - temp (number): optional 
              - illum (number): optional 

            - specify a list of the exponents on the 7 fundamental properties:
              - dim (tuple): (length, time, mass, angle, charge, temp, illum)

            - copy constructor
              - dim (Dim): existing Dim object
        """
        arg_exp = (length,time,mass,angle,charge,temp,illum)
        if dim is not None:
            if np.any(arg_exp):
                raise TypeError("Cannot specify both the dim tuple and individual exponents")
            if type(dim) is Dim:
                # assume that if dim is already a Dim, we don't need to check the exponent types
                exp = dim._exp
            else:
                # Try to convert it to a tuple and verify that the tuple has 7 entries
                try:
                    exp = tuple(dim)
                    assert len(exp) == 7
                except:
                    raise NotDimLike(dim)
                # Verify that all 7 entries are floating point numbers (and not complex)
                if not np.all([isinstance(x,numbers.Real) for x in dim]):
                    raise TypeError(f"Exponents on dimensions must all be real numbers, not {dim}")
        else:
            # dim not specified, so we're using the kwargs arguments ( after verifying them )
            keys = ('length','time','mass','angle','charge','temp','illum')
            for k,v in zip(keys,arg_exp):
                if not isinstance(v,numbers.Real):
                    raise TypeError(f"Exponent on {k} must be a real number, not {v}")
            exp = arg_exp 

        self._exp = exp


    @property
    def is_dimensionless(self):
        return not np.any(self._exp)

    @property
    def is_angle(self):
        return self._exp in ((0,0,0,1,0,0,0),(0,0,0,0,0,0,0))

    @property
    def inverse(self):
        return Dim(tuple(-t for t in self._exp))

    def __eq__(self,other):
        try:
            return self._exp == other._exp
        except:
            return False

    def __mul__(self,other):
        try:
            return Dim(tuple(a+b for a,b in zip(self._exp,other._exp)))
        except:
            raise NotDimLike(other)

    __rmul__ = __mul__

    def __truediv__(self,other):
        try:
            return Dim(tuple(a-b for a,b in zip(self._exp,other._exp)))
        except:
            raise NotDimLike(other)

    def __rtruediv__(self,other):
        try:
            return Dim(tuple(b-a for a,b in zip(self._exp,other._exp)))
        except:
            raise NotDimLike(other)

    def __pow__(self,n):
        return Dim(tuple(a*n for a in self._exp))

    def __repr__(self):
        keys = ('length','time','mass','angle','charge','temp','illum')
        return (f"Dim("
                + ",".join(f"{n}={e}" for n,e in zip(keys,self._exp) if e)
                + ",)")

    def __str__(self):
        if self._exp in _DERIVED_DIMS:
            return _DERIVED_DIMS[self._exp]

        names = ('L','T','M','A','C','K','I')
        num = " ".join(f"{n}" if e==1 else f"{n}^{e}" for n,e in
                       zip(names,self._exp) if e>0)
        den = " ".join(f"{n}" if e==-1 else f"{n}^{-e}" for n,e in
                       zip(names,self._exp) if e<0)

        if num:
            if den:
                return f"[{num}/{den}]"
            else:
                return f"[{num}]"
        elif den:
            return f"[1/{den}]"
        else:
            return f"[]"
        



