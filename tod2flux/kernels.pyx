#cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=True, cdivision=True

""" Cython kernels to speed up processing """

from cython.parallel import prange
import numpy as np

cimport numpy as np
cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas
from libc.math cimport sqrt, fabs, sin, cos, M_PI
cimport cython

