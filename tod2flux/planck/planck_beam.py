import os
import sys

import astropy.io.fits as pf
import numpy as np
from scipy.constants import degree, arcmin
import scipy.interpolate

from .. import Beam


DATADIR = os.path.join(os.path.dirname(__file__), "data")

FREQ2FWHM = {
    30: 32.7,
    44: 27.9,
    70: 13.0,
    100: 9.5,
    143: 7.0,
    217: 4.7,
    353: 4.5,
    545: 4.7,
    857: 4.4,
}


def parse_gridded_beam(
    beamfile,
    ihdu=1,
    coord_base=0,
    pol=False,
    nx=None,
    ny=None,
    xcenter=None,
    ycenter=None,
    xdelta=None,
    ydelta=None,
):
    """
    Parse the contents of a fits file containing a square beam map

    Inputs:
    beamfile -- beam file name
    ihdu (1) -- fits extension to load the beam from
    coord_base (0) -- indexing convention 0 (C) or 1 (Fortran) based
    """
    print("Reading gridded beam from " + beamfile)

    if coord_base == None:
        coord_base = 0

    if coord_base not in [0, 1]:
        raise Exception("Unsupported coord_base: " + str(coord_base))

    try:
        hdulist = pf.open(beamfile)
    except OSError as e:
        print("Failed to open {} : '{}'".format(beamfile, e))
        raise

    header = hdulist[ihdu].header

    if "baseidx" in header:
        coord_base = header["BASEIDX"]

    print("Coord_base : ", coord_base)

    if not nx:
        if "nx" in header:
            nx = header["NX"]
        else:
            print("Warning: Bypassing bad HIERARCH card by explicit indexing")
            import pdb

            pdb.set_trace()
            nx = header[22]
            ny = header[23]
            xcenter = header[24]
            ycenter = header[25]
            xdelta = header[26] / arcmin
            ydelta = header[27] / arcmin

    if not ny:
        ny = header["NY"]

    if not xcenter:
        if "XCENTRE" in header:
            xcenter = header["XCENTRE"]
        elif "XCENTER" in header:
            xcenter = header["XCENTER"]
        else:
            raise Exception("{} does not specifify grid center coordinate")
    if not ycenter:
        if "YCENTRE" in header:
            ycenter = header["YCENTRE"]
        elif "XCENTER" in header:
            ycenter = header["YCENTER"]
        else:
            raise Exception("{} does not specifify grid center coordinate")

    if not xdelta:
        xdelta = header["XDELTA"] / arcmin
    if not ydelta:
        ydelta = header["YDELTA"] / arcmin

    if "fwhm" in header:
        eg = []  # elliptical gaussian params
        eg.append(header["fwhm"])
        eg.append(header["ellipticity"])
        eg.append(header["psi_ell"])
    else:
        eg = None

    grbeam_x = (np.arange(nx) - xcenter + coord_base) * xdelta
    grbeam_y = (np.arange(ny) - ycenter + coord_base) * ydelta

    grbeam = np.reshape(hdulist[ihdu].data.field(0).ravel(), [ny, nx])
    if pol:
        grbeam_Q = np.reshape(hdulist[ihdu].data.field(1), [ny, nx])
        grbeam_U = np.reshape(hdulist[ihdu].data.field(2), [ny, nx])
    else:
        grbeam_Q = None
        grbeam_U = None
    beammax = np.amax(grbeam)
    # if np.abs(beammax - 1) > 1.1:
    grbeam /= beammax

    hdulist.close()

    """
    while nx > 800:
        # Cannot interpolate too large grid
        print("Pruning the grid ...")
        ind = slice(0, nx, 2)
        grbeam_x = grbeam_x[ind]
        grbeam_y = grbeam_y[ind]
        grbeam = grbeam[ind, ind]
        if pol:
            grbeam_Q = grbeam_Q[ind, ind]
            grbeam_U = grbeam_U[ind, ind]
        nx, ny = len(grbeam_x), len(grbeam_y)
    """

    xdelta = (grbeam_x[-1] - grbeam_x[0]) / (nx - 1)
    ydelta = (grbeam_y[-1] - grbeam_y[0]) / (ny - 1)

    beam_solid_angle = np.sum(grbeam) * xdelta * ydelta

    # construct a linear interpolator on the map

    print("Constructing linear interpolator for the {} x {} beam map".format(nx, ny))
    grbeam_interp = scipy.interpolate.RectBivariateSpline(
        grbeam_x, grbeam_y, grbeam.T, kx=1, ky=1
    )

    if pol:
        grbeam_interp_Q = scipy.interpolate.RectBivariateSpline(
            grbeam_x, grbeam_y, grbeam_Q.T, kx=1, ky=1
        )
        grbeam_interp_U = scipy.interpolate.RectBivariateSpline(
            grbeam_x, grbeam_y, grbeam_U.T, kx=1, ky=1
        )
    else:
        grbeam_interp_Q = None
        grbeam_interp_U = None

    return (
        nx,
        ny,
        xcenter,
        ycenter,
        xdelta,
        ydelta,
        grbeam_x,
        grbeam_y,
        grbeam,
        grbeam_Q,
        grbeam_U,
        beam_solid_angle,
        grbeam_interp,
        grbeam_interp_Q,
        grbeam_interp_U,
        eg,
    )


def parse_polar_gridded_beam(
    beamfile,
    ihdu=1,
    pol=False,
    nx=1201,
    ny=1201,
    xcenter=600,
    ycenter=600,
    xdelta=None,
    ydelta=None,
    coord_base=0,
):
    """
    Parse the contents of a fits file containing a circular beam map

    Inputs:
    beamfile -- beam file name
    ihdu (1) -- fits extension to load the beam from
    coord_base (0) -- indexing convention 0 (C) or 1 (Fortran) based
    """
    print("Reading polar gridded beam from " + beamfile)

    try:
        hdulist = pf.open(beamfile)
    except OSError as e:
        print("Failed to open {} : '{}'".format(beamfile, e))
        raise

    header = hdulist[ihdu].header

    print("Coord_base : ", coord_base)

    ntheta = header["Ntheta"]
    nphi = header["Nphi"]
    theta_min = header["Mintheta"]
    theta_max = header["Maxtheta"]
    phi_step = 2 * np.pi / (nphi - 1)
    theta_step = (theta_max - theta_min) / (ntheta - 1)

    # Find the largest square grid that will fit the circular map

    rmax = theta_max / np.sqrt(2)

    xdelta = 2 * rmax / (nx - 1)
    ydelta = 2 * rmax / (ny - 1)

    eg = None

    grbeam_x = (np.arange(nx) - xcenter + coord_base) * xdelta
    grbeam_y = (np.arange(ny) - ycenter + coord_base) * ydelta

    polarbeam = np.reshape(hdulist[ihdu].data.field(0).ravel(), [ntheta, nphi])
    if pol:
        polarbeam_Q = np.reshape(hdulist[ihdu].data.field(1).ravel(), [ntheta, nphi])
        polarbeam_U = np.reshape(hdulist[ihdu].data.field(2).ravel(), [ntheta, nphi])
    beammax = np.amax(polarbeam)
    polarbeam /= beammax

    hdulist.close()

    # Now sample the beam onto a square grid

    xgrid, ygrid = np.meshgrid(grbeam_x, grbeam_y)
    grbeam_theta = np.sqrt(xgrid ** 2 + ygrid ** 2)
    grbeam_phi = np.arctan2(ygrid, xgrid) % (2 * np.pi)
    itheta = ((grbeam_theta - theta_min) / theta_step).astype(np.int)
    itheta[itheta > ntheta - 1] = ntheta - 1
    iphi = (grbeam_phi / phi_step).astype(np.int)

    grbeam = polarbeam[itheta, iphi].copy()
    if pol:
        grbeam_Q = polarbeam_Q[itheta, iphi].copy()
        grbeam_U = polarbeam_U[itheta, iphi].copy()
    else:
        grbeam_Q = None
        grbeam_U = None

    # Convert the steps to arcmin

    xdelta /= arcmin
    ydelta /= arcmin

    beam_solid_angle = np.sum(grbeam) * xdelta * ydelta

    # construct a linear interpolator on the map

    points = np.array([xgrid.flatten(), ygrid.flatten()]).T

    print("Constructing linear interpolator for the {} x {} beam map".format(nx, ny))
    grbeam_interp = scipy.interpolate.RectBivariateSpline(
        grbeam_x / arcmin, grbeam_y / arcmin, grbeam.T, kx=1, ky=1
    )

    if pol:
        grbeam_interp_Q = scipy.interpolate.RectBivariateSpline(
            grbeam_x / arcmin, grbeam_y / arcmin, grbeam_Q.T, kx=1, ky=1
        )
        grbeam_interp_U = scipy.interpolate.RectBivariateSpline(
            grbeam_x / arcmin, grbeam_y / arcmin, grbeam_U.T, kx=1, ky=1
        )
    else:
        grbeam_interp_Q = None
        grbeam_interp_U = None

    return (
        nx,
        ny,
        xcenter,
        ycenter,
        xdelta,
        ydelta,
        grbeam_x,
        grbeam_y,
        grbeam,
        grbeam_Q,
        grbeam_U,
        beam_solid_angle,
        grbeam_interp,
        grbeam_interp_Q,
        grbeam_interp_U,
        eg,
    )


class PlanckBeam(Beam):
    def __init__(self, detector_name, psi_uv, epsilon, fwhm_arcmin, pol=False):
        self.detector_name = detector_name
        self.psi_uv = psi_uv
        self.epsilon = epsilon
        self.pol = pol
        self.referential = "Dxx"  # interface is always Dxx
        self.xsign = 1  # PSF is directly the source image, no reflections needed
        self.ysign = 1  # Scan is from bottom to top
        self.xsign_pxx = 1
        self.ysign_pxx = 1
        self.r = -1
        self.offset = None
        self.base_index = 0
        self.fwhm_arcmin = fwhm_arcmin
        self._solid_angle = None
        if "LFI" in detector_name:
            self.load_lfi_beam()
        else:
            self.load_hfi_beam()
        return

    def get_gr_dxx_beam(self, x, y, pol=False, grid=False):
        """
        Return the value of the gridded beam in the Dxx coordinate system.

        Underlying implementation is linear interpolation on a fine grid.
        
        Inputs:
        x, y -- arrays of the same shape with coordinates in arc minutes
        """
        if np.shape(x) != np.shape(y):
            print("ERROR in get_beam: shapes do not match")
            exit(-1)

        points = np.array([self.xsign * x, self.ysign * y]).T

        if pol:
            return np.vstack(
                [
                    self.grbeam_interp(x, y, grid=grid),
                    self.grbeam_interp_Q(x, y, grid=grid),
                    self.grbeam_interp_U(x, y, grid=grid),
                ]
            )
        else:
            return self.grbeam_interp(x, y, grid=grid)

    def get_gr_pxx_beam(self, x, y, pol=False, grid=False):
        """
        Sample the interpolated square map at (x, y)
        """
        if np.shape(x) != np.shape(y):
            print("ERROR in get_beam: shapes do not match")
            exit(-1)

        xtemp = self.xsign_pxx * x
        ytemp = self.ysign_pxx * y

        psi_pol = self.psi_uv
        # if self.rotate_beam:
        #    psi_pol += self.rotate_beam

        xx = xtemp * np.cos(psi_pol * degree) + ytemp * np.sin(psi_pol * degree)
        yy = -xtemp * np.sin(psi_pol * degree) + ytemp * np.cos(psi_pol * degree)

        return self.get_gr_dxx_beam(xx, yy, pol=pol, grid=grid)

    def load_hfi_beam(self):
        self.instrument = "HFI"
        self.freq = float(self.detector_name[0:3])
        # To be replaced later with more reliable estimates
        self.fwhm = FREQ2FWHM[self.freq]
        self.horn = int(self.detector_name[4:5])
        self.psi_uv = 0.0

        self.xsign *= -1
        self.ysign *= -1
        self.base_index = 1  # Either 0 or 1

        beamfile = os.path.join(
            DATADIR,
            "BS_HBM_DX11v67_I5_HIGHRES_POLAR_{}_xp.fits".format(self.detector_name),
        )

        if not os.path.isfile(beamfile):
            raise Exception("ERROR: beamfile, {}, does not exist!".format(beamfile))

        print("Reading gridded beam from " + beamfile)

        (
            self.nx,
            self.ny,
            self.xcenter,
            self.ycenter,
            self.xdelta,
            self.ydelta,
            self.grbeam_x,
            self.grbeam_y,
            self.grbeam,
            self.grbeam_Q,
            self.grbeam_U,
            self.beam_solid_angle,
            self.grbeam_interp,
            self.grbeam_interp_Q,
            self.grbeam_interp_U,
            eg,
        ) = parse_polar_gridded_beam(beamfile, coord_base=self.base_index, pol=False)

        if eg != None:
            self.fwhm, self.ellipticity, self.psi_ell = eg

        self.get_beam = self.get_gr_dxx_beam
        return

    def load_lfi_beam(self):
        self.instrument = "LFI"
        self.horn = int(self.detector_name[3:5])
        if self.horn < 24:
            self.freq = 70.0
        elif self.horn < 27:
            self.freq = 44.0
        else:
            self.freq = 30.0
        # To be replaced later with more reliable estimates
        self.fwhm = FREQ2FWHM[self.freq]

        self.arm = self.detector_name[-1]
        if self.arm == "M":
            self.arm2 = "y"
        else:
            self.arm2 = "x"

        self.xsign_pxx = -1
        self.ysign_pxx = -1

        beamfile = os.path.join(
            DATADIR,
            "mb_lfi_{}_{}_{}_qucs-raa_dBdTcmb.stokes".format(
                int(self.freq), int(self.horn), self.arm2
            ),
        )

        if not os.path.isfile(beamfile):
            raise Exception("ERROR: beamfile, {}, does not exist!".format(beamfile))

        (
            self.nx,
            self.ny,
            self.xcenter,
            self.ycenter,
            self.xdelta,
            self.ydelta,
            self.grbeam_x,
            self.grbeam_y,
            self.grbeam,
            self.grbeam_Q,
            self.grbeam_U,
            self.beam_solid_angle,
            self.grbeam_interp,
            self.grbeam_interp_Q,
            self.grbeam_interp_U,
            eg,
        ) = parse_gridded_beam(beamfile, pol=self.pol, coord_base=self.base_index)

        if eg != None:
            self.fwhm, self.ellipticity, self.psi_ell = eg

        self.get_beam = self.get_gr_pxx_beam

        return

    def get_eg_beam(self, x, y):
        """
        Return the elliptical Gaussian beam based on the stored gaussian parameters
        """
        if np.shape(x) != np.shape(y):
            print("ERROR in get_beam: shapes do not match")
            exit(-1)

        xx = x * np.cos(self.psi_ell * degree) + y * np.sin(self.psi_ell * degree)
        yy = -x * np.sin(self.psi_ell * degree) + y * np.cos(self.psi_ell * degree)

        return np.exp(
            -0.5 * (xx ** 2 / self.sigma_x ** 2 + yy ** 2 / self.sigma_y ** 2)
        )

    def bsa(self):
        """ Calculate the beam solid angle
        """
        r = None
        if self.r > 0:
            r = self.r
        self.get_map(r)

        delta = self.grid[1] - self.grid[0]

        return np.sum(self.map * delta ** 2)

    def geometric_fwhm(self):
        return np.sqrt(self.beam_solid_angle * 4 * np.log(2) / np.pi)

    def get_psf(self, x, y):
        """
        Return the point spread function
        
        Inputs:
        x, y -- arrays of same shape, containing the coordinates in arc minutes
        """
        # get_beam is defined upon initialization
        return self.get_beam(-x, -y)

    def get_beam_buffered(self, x, y, pol=False):
        """
        Evaluate the beam using buffers to reduce the memory requirement

        Inputs:
        x, y -- arrays of same shape, containing the coordinates in arc minutes
        """

        xx = x.flatten()
        yy = y.flatten()

        n = len(xx)
        zz = np.zeros(n)
        if pol:
            zz_Q = np.zeros(n)
            zz_U = np.zeros(n)

        nbuf = 10000

        for i in range(0, n, nbuf):
            i2 = np.amin([i + nbuf, n])
            if pol:
                zz[i:i2], zz_Q[i:i2], zz_U[i:i2] = self.get_beam(
                    xx[i:i2], yy[i:i2], None, pol=pol
                )
            else:
                zz[i:i2] = self.get_beam(xx[i:i2], yy[i:i2], None)

        if pol:
            return zz, zz_Q, zz_U
        else:
            return zz

    def get_map(self, r, offset=None, pol=False):
        """
        Set the members x, y and map to image of the beam.

        Inputs:
        r -- radius of the grid, default 2.5 FWHM
        offset -- optional coordinate offset added to the beam center
        """
        if r == None:
            r = 2.5 * self.fwhm

        if r == self.r and offset == None:
            if not pol or self.map_Q != None:
                # Already evaluated
                return

        self.r = r
        self.offset = offset

        self.npix = 1000

        self.grid = np.linspace(-self.r, self.r, self.npix)
        self.x, self.y = np.meshgrid(self.grid, self.grid)

        psi_uv_orig = self.psi_uv
        if offset != None:
            if len(offset) < 2:
                raise Exception("offset length must be at least 2")
            xoff, yoff = offset[0:2]
            if len(offset) > 2:
                dpsi = offset[2]
                self.psi_uv += dpsi
        else:
            xoff, yoff = 0.0, 0.0

        if pol:
            maps = self.get_beam_buffered(self.x - xoff, self.y - yoff, None, pol=pol)
            self.map = np.reshape(maps[0], [self.npix, self.npix])
            self.map_Q = np.reshape(maps[1], [self.npix, self.npix])
            self.map_U = np.reshape(maps[2], [self.npix, self.npix])

            amp = np.amax(self.map)
            self.map /= amp
            self.map_Q /= amp
            self.map_U /= amp
        else:
            self.map = np.reshape(
                self.get_beam_buffered(self.x - xoff, self.y - yoff, None),
                [self.npix, self.npix],
            )
            self.map /= np.amax(self.map)

        self.psi_uv = psi_uv_orig

    def get_barycenter(self, r=None):

        self.get_map(r, pol=False)

        norm = np.sum(self.map)
        x0 = np.sum(self.x * self.map) / norm
        y0 = np.sum(self.y * self.map) / norm

        return [x0, y0]

    def fit_eg(self, r=None):

        self.get_map(r, pol=False)

        def residuals(p, x, y, z):
            x0, y0, sigma_x, sigma_y, psi_ell = p

            xx = (x - x0) * np.cos(psi_ell) + (y - y0) * np.sin(psi_ell)
            yy = -(x - x0) * np.sin(psi_ell) + (y - y0) * np.cos(psi_ell)

            return z - np.exp(-0.5 * (xx ** 2 / sigma_x ** 2 + yy ** 2 / sigma_y ** 2))

        p0 = [0, 0, self.fwhm, self.fwhm, 0]
        p, bopt, info, msg, ier = scipy.optimize.leastsq(
            residuals,
            p0,
            args=(self.x.ravel(), self.y.ravel(), self.map.ravel()),
            full_output=True,
            Dfun=None,
            maxfev=10000,
        )

        if ier not in range(5):
            print("ier == ", ier)
            print("msg == ", msg)

        x0, y0, sigma_x, sigma_y, psi_ell = p

        print("Best fit elliptical gaussian:")
        print("x0 = {:.3g}''".format(x0 * 60.0))
        print("y0 = {:.3g}''".format(y0 * 60.0))
        print("sigma_x = {:.3g}'".format(sigma_x))
        print("sigma_y = {:.3g}'".format(sigma_y))
        print("psi_ell = {:.3g} deg".format(psi_ell / degree % 180.0))

        return p

    @property
    def solid_angle(self):
        """ Return the beam solid angle
        """
        if self._solid_angle is None:
            self._solid_angle = self.bsa()
        return self._solid_angle

    @property
    def solid_angle_err(self):
        """ Return the beam solid angle error
        """
        return 0.0
