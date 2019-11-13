import time

import astropy.io.fits as pf
import numpy as np
from scipy.constants import degree, arcmin


class Dataset:
    """ A class representing a small dataset pertaining to one detector
    and one target.
    """
    
    def __init__(self, filename, psi_pol=None):
        self.filename = filename

        print("Reading {} ... ".format(filename))
        t1 = time.time()

        hdulist = pf.open(filename)
        t2 = time.time()
        print("Done in {:6.2f} s".format(t2 - t1))

        detector = hdulist[1].header["DETECTOR"]

        self.target = hdulist[1].header["TARGET"]
        self.info = hdulist[1].header["INFO"]
        self.radius = hdulist[1].header["RADIUS"]
        self.target_lon = hdulist[1].header["LON"]
        self.target_lat = hdulist[1].header["LAT"]
        self.target_phi = self.target_lon * degree
        self.target_theta = (90 - self.target_lat) * degree
        if "SIGMA" in hdulist[1].columns.names:
            self.sigma = hdulist[1].header["SIGMA"] * 1e3
        else:
            self.sigma = None
        self.fsample = hdulist[1].header["FSAMPLE"]

        self.time = hdulist[1].data.field("TIME").flatten()
        self.theta = hdulist[1].data.field("THETA").flatten()
        self.phi = hdulist[1].data.field("PHI").flatten()
        self.psi = hdulist[1].data.field("PSI").flatten()
        self.signal = hdulist[1].data.field("SIGNAL").flatten()

        print("Concatenating the arrays ...")
        for i in range(2, len(hdulist)):
            self.time = np.append(self.time, hdulist[i].data.field("TIME").flatten())
            self.theta = np.append(self.theta, hdulist[i].data.field("THETA").flatten())
            self.phi = np.append(self.phi, hdulist[i].data.field("PHI").flatten())
            self.psi = np.append(self.psi, hdulist[i].data.field("PSI").flatten())
            self.signal = np.append(self.signal, hdulist[i].data.field("SIGNAL").flatten())

        # Remove possible zero padding in the arrays
        ind = self.time != 0
        if np.sum(ind) < len(ind):
            self.time = self.time[ind]
            self.theta = self.theta[ind]
            self.phi = self.phi[ind]
            self.psi = self.psi[ind]
            self.signal = self.signal[ind]

        if psi_pol is not None:
            # remove polarizer angle from psi
            self.psi -= (psi_pol + 180) * degree

        """
        # Correct for a bug in the HFI pointing library

        if detector == "100-4a":
            psi -= 90 * degree
        if detector == "100-4b":
            psi += 90 * degree
        """

        # To mK !!!
        self.signal *= 1e3

        self.size = self.signal.size

        hdulist.close()
        
        return
