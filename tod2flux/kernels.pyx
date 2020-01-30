#cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=True, cdivision=True

""" Cython kernels to speed up processing """

from cython.parallel import prange
import numpy as np

cimport numpy as np
cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas
from libc.math cimport sqrt, fabs, sin, cos, M_PI
cimport cython


def fast_bin_map(
        double[:] x not None,
        double[:] y not None,
        double[:] z not None,
        double wbin,
        double radius,
):
    cdef long n = len(z)
    cdef long i, xbin, ybin
    cdef long nbin = <long> (2. * radius / wbin + 1.)
    cdef double wbininv = 1. / wbin

    cdef np.ndarray sigmap = np.zeros([nbin, nbin], dtype=np.float64)
    cdef np.ndarray hitmap = np.zeros([nbin, nbin], dtype=np.int64)
    cdef double[:, :] sigmap_view = sigmap
    cdef long[:, :] hitmap_view = hitmap

    for i in range(n):
        if fabs(x[i]) < radius and fabs(y[i]) < radius:
            xbin = <long> ((x[i] + radius) * wbininv)
            ybin = <long> ((y[i] + radius) * wbininv)
            hitmap_view[xbin, ybin] += 1
            sigmap_view[xbin, ybin] += z[i]

    for xbin in range(nbin):
        for ybin in range(nbin):
            if hitmap_view[xbin, ybin] != 0:
                sigmap_view[xbin, ybin] /= hitmap_view[xbin, ybin]

    return sigmap, hitmap
