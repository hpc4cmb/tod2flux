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


DETECTOR_SETS = {
    "030": ["LFI27M", "LFI27S", "LFI28M", "LFI28S"],
    "044-24": ["LFI24M", "LFI24S"],
    "044-25/26": ["LFI25M", "LFI25S", "LFI26M", "LFI26S"],
    "070-18/23": ["LFI18M", "LFI18S", "LFI23M", "LFI23S"],
    "070-19/22": ["LFI19M", "LFI19S", "LFI22M", "LFI22S"],
    "070-20/21": ["LFI20M", "LFI20S", "LFI21M", "LFI21S"],
    "100-1/4": ["100-1a", "100-1b", "100-4a", "100-4b"],
    "100-2/3": ["100-2a", "100-2b", "100-3a", "100-3b"],
    "143-1/3": ["143-1a", "143-1b", "143-3a", "143-3b",],
    "143-2/4": ["143-2a", "143-2b", "143-4a", "143-4b",],
    "143-swb": ["143-5", "143-6", "143-7"],
    "217-5/7": ["217-5a", "217-5b", "217-7a", "217-7b",],
    "217-6/8": ["217-6a", "217-6b", "217-8a", "217-8b",],
    "217-swb": ["217-1", "217-2", "217-3", "217-4"],
    "353-3/5": ["353-3a", "353-3b", "353-5a", "353-5b",],
    "353-4/6": ["353-4a", "353-4b", "353-6a", "353-6b",],
    "353-swb": ["353-1", "353-2", "353-7", "353-8"],
    "545": ["545-1", "545-2", "545-4"],
    "857": ["857-1", "857-2", "857-3", "857-4"],
}


def parse_lfi_rimo(rimofile, detector):
    print("Reading the LFI RIMO from {}".format(rimofile))

    try:
        rimo = pf.open(rimofile)
    except OSError as e:
        print("Failed to open {} : '{}'".format(rimofile, e))
        raise
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

    try:
        rimo = pf.open(rimofile)
    except OSError as e:
        print("Failed to open {} : '{}'".format(rimofile, e))
        raise
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
        self._detector_set = None
        for detset_name, detset in DETECTOR_SETS.items():
            if name in detset:
                if self._detector_set is not None:
                    raise RuntimeError(
                        "{} is in multiple detector sets: {}, {}".format(
                            name, self._detector_set, detset_name
                        )
                    )
                self._detector_set = detset_name
                break
        if self._detector_set is None:
            raise RuntimeError("{} is not in any detector sets".format(name))
        return

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

    @property
    def detector_set(self):
        return self._detector_set
