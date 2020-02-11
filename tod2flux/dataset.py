import os
import sys
import time

import astropy.io.fits as pf
import numpy as np
from scipy.constants import degree, arcmin


class Dataset:
    """ A class representing a small dataset pertaining to one detector
    and one target.
    """

    def __init__(self, filename, time_offset=None, time_scale=None):
        """ Instantiate a Dataset object.  The time stamps should be in
        UNIX time.  If the dataset time is in different units, it can
        be modified with time_offset and time_scale.

        Args:
        filename(str) :  Full path to the small dataset FITS file
        time_offset(float) : offset to subtract from the time stamps
        time_scale(float) : scaling to apply to the time stamps
        """
        self.filename = filename
        self.name = os.path.basename(filename)

        print("Reading {} ... ".format(filename), flush=True)
        t1 = time.time()

        try:
            hdulist = pf.open(filename)
        except OSError as e:
            print("Failed to open {} : '{}'".format(filename, e))
            raise
        t2 = time.time()
        print("Done in {:6.2f} s".format(t2 - t1), flush=True)

        self.detector = hdulist[1].header["DETECTOR"]
        self.target = hdulist[1].header["TARGET"]
        self.info = hdulist[1].header["INFO"]
        self.radius_arcmin = hdulist[1].header["RADIUS"]
        self.target_lon_deg = hdulist[1].header["LON"]
        self.target_lat_deg = hdulist[1].header["LAT"]
        self.target_phi = self.target_lon_deg * degree
        self.target_theta = (90 - self.target_lat_deg) * degree
        try:
            self.coord = hdulist[1].header["COORD"]
        except Exception as e:
            print(
                "WARNING: could not read coordinate system from '{}': '{}'. "
                "Defaulting to GALACTIC".format(filename, e)
            )
            self.coord = "G"
        if "SIGMA" in hdulist[1].columns.names:
            self.sigma_mKCMB = hdulist[1].header["SIGMA"] * 1e3
        else:
            self.sigma_mKCMB = None
        self.fsample = hdulist[1].header["FSAMPLE"]

        self.time_s = hdulist[1].data.field("TIME").flatten()
        if time_offset is not None:
            self.time_s += time_offset
        if time_scale is not None:
            self.time_s *= time_scale
        self.theta = hdulist[1].data.field("THETA").flatten()
        self.phi = hdulist[1].data.field("PHI").flatten()
        self.psi = hdulist[1].data.field("PSI").flatten()
        self.signal_mK = hdulist[1].data.field("SIGNAL").flatten()

        print("Concatenating the arrays ...", flush=True)
        for i in range(2, len(hdulist)):
            self.time_s = np.append(
                self.time_s, hdulist[i].data.field("TIME").flatten()
            )
            self.theta = np.append(self.theta, hdulist[i].data.field("THETA").flatten())
            self.phi = np.append(self.phi, hdulist[i].data.field("PHI").flatten())
            self.psi = np.append(self.psi, hdulist[i].data.field("PSI").flatten())
            self.signal_mK = np.append(
                self.signal_mK, hdulist[i].data.field("SIGNAL").flatten()
            )

        # Remove possible zero padding in the arrays
        ind = self.time_s != 0
        if np.sum(ind) < len(ind):
            self.time_s = self.time_s[ind]
            self.theta = self.theta[ind]
            self.phi = self.phi[ind]
            self.psi = self.psi[ind]
            self.signal_mK = self.signal_mK[ind]

        """
        # Correct for a bug in the HFI pointing library

        if detector == "100-4a":
            psi -= 90 * degree
        if detector == "100-4b":
            psi += 90 * degree
        """

        # To mK !!!
        self.signal_mK *= 1e3

        self.size = self.signal_mK.size

        hdulist.close()

        return

    def __str__(self):
        result = "small dataset:\n"
        result += "  Target = '{}'\n".format(self.target)
        result += "  Info = '{}'\n".format(self.info)
        result += "  (lon, lat) = ({}, {}) degrees\n".format(
            self.target_lon_deg, self.target_lat_deg
        )
        result += "  Search radius = {}'\n".format(self.radius_arcmin)
        result += "  total time = {:5.2f} min\n".format(self.size / self.fsample / 60)
        result += "  Detector = '{}'\n".format(self.detector)
        result += "  sigma = {} mK\n".format(self.sigma_mKCMB)
        """
        print("      psi_ell == {:5.3f} deg".format(psi_ell))
        print("         FWHM == {:5.3f}'".format(fwhm0))
        print("  solid angle == {:5.3f} sq arc min".format(bsa0))
        print("  ellipticity == {:5.3f}".format(ellipticity0))
        print(" MJy / mK_CMB == {:.4g}".format(mkcmb2mjysr))
        """
        return result
