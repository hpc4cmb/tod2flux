import os
import sys

import astropy.io.fits as pf
import numpy as np

from .. import Detector

from .planck_beam import PlanckBeam
from .planck_bandpass import PlanckBandpass


DATADIR = os.path.join(os.path.dirname(__file__), "data")
RIMOFILE_LFI = os.path.join(DATADIR, "RIMO_LFI.fits")
RIMOFILE_HFI = os.path.join(DATADIR, "RIMO_HFI.fits")


def parse_lfi_rimo(rimofile, detector):
    print("Reading the LFI RIMO from {}".format(rimofile))

    rimo = pf.open(rimofile)
    ind = np.array(rimo[1].data.field("detector").flatten()) == detector
    if sum(ind) == 0:
        ind = np.array(rimo[1].data.field("detector").flatten()) == "{:8}".format(
            detector
        )
    if sum(ind) == 0:
        raise Exception("Could not find {} in {}".format(detector, rimofile))
    psi_uv = rimo[1].data[ind]["PSI_UV"][0]
    psi_pol = rimo[1].data[ind]["PSI_POL"][0] + psi_uv
    fwhm = rimo[1].data[ind]["FWHM"][0]
    ellipticity = rimo[1].data[ind]["ELLIPTICITY"][0]
    psi_ell = rimo[1].data[ind]["POSANG"][0] + psi_uv
    fsample = rimo[1].data[ind]["F_SAMP"][0]
    net = rimo[1].data[ind]["NET"][0]
    sigma = net * np.sqrt(fsample)
    # The RIMO does not contain nominal frequencies ...
    horn = int(detector[3:5])
    if horn in range(27, 29):
        frequency = 30
    elif horn in range(24, 27):
        frequency = 44
    elif horn in range(18, 24):
        frequency = 70
    else:
        raise RuntimeError("Cannot determine frequency for {}".format(detector))
    rimo.close()

    return psi_uv, psi_pol, fwhm, ellipticity, psi_ell, fsample, sigma, frequency


def parse_hfi_rimo(rimofile, detector):
    print("Reading the HFI RIMO from {}".format(rimofile))

    rimo = pf.open(rimofile)
    ind = np.array(rimo[1].data.field("detector").flatten()) == detector
    if np.sum(ind) == 0:
        print("ERROR: {} does not match any of:".format(detector))
        print(rimo[1].data.field("detector").flatten())
        return None
    psi_uv = rimo[1].data[ind]["PSI_UV"][0]
    psi_pol = rimo[1].data[ind]["PSI_POL"][0] + psi_uv
    fwhm = rimo[1].data[ind]["FWHM"][0]
    ellipticity = rimo[1].data[ind]["ELLIPTICITY"][0]
    psi_ell = rimo[1].data[ind]["POSANG"][0] + psi_uv
    fsample = rimo[1].data[ind]["F_SAMP"][0]
    epsilon = rimo[1].data[ind]["EPSILON"][0]
    net = rimo[1].data[ind]["NET"][0]
    sigma = net * np.sqrt(fsample)
    # The RIMO does not contain nominal frequencies ...
    frequency = int(detector[:3])

    return (
        psi_uv,
        psi_pol,
        fwhm,
        ellipticity,
        psi_ell,
        fsample,
        epsilon,
        sigma,
        frequency,
    )


class PlanckDetector(Detector):
    def __init__(self, name):
        self.name = name
        if "LFI" in name:
            self.rimofile = RIMOFILE_LFI
            self.epsilon = 0
            (
                self.psi_uv_deg,
                self._psi_pol_deg,
                self.fwhm_arcmin,
                self.ellipticity,
                self.psi_ell_deg,
                self._fsample,
                self._sigma_KCMB,
                self.frequency,
            ) = parse_lfi_rimo(self.rimofile, name)
        else:
            self.rimofile = RIMOFILE_HFI
            (
                self.psi_uv_deg,
                self._psi_pol_deg,
                self.fwhm_arcmin,
                self.ellipticity,
                self.psi_ell_deg,
                self._fsample,
                self.epsilon,
                self._sigma_KCMB,
                self.frequency,
            ) = parse_hfi_rimo(self.rimofile, name)
        self._beam = PlanckBeam(name, self.psi_uv_deg, self.epsilon, self.fwhm_arcmin)
        self._bandpass = PlanckBandpass(name)

    @property
    def beam(self):
        return self._beam

    @property
    def bandpass(self):
        return self._bandpass

    @property
    def psi_pol_deg(self):
        return self._psi_pol_deg

    @property
    def sigma_KCMB(self):
        return self._sigma_KCMB

    @property
    def fsample(self):
        return self._fsample

    @property
    def pol_efficiency(self):
        return (1 - self.epsilon) / (1 + self.epsilon)

    @property
    def nominal_frequency(self):
        return self.frequency
