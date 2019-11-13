
import astropy.io.fits as pf

from .. import Detector

from .planck_beam import PlanckBeam
from .planck_bandpass import PlanckBandpass


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
    rimo.close()

    return psi_uv, psi_pol, fwhm, ellipticity, psi_ell, fsample, sigma


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
    rimo.close()

    return psi_uv, psi_pol, fwhm, ellipticity, psi_ell, fsample, epsilon, sigma


class PlanckDetector(Detector):
    
    def __init__(self, name):
        self.name = name
        if "LFI" in name:
            self.rimofile = "RIMO_LFI_npipe5_symmetrized.fits"
            self.epsilon = 0
            (
                self.psi_uv,
                self.psi_pol,
                self.fwhm,
                self.ellipticity,
                self.psi_ell,
                self.fsample,
                self.sigma,
            ) = parse_lfi_rimo(self.rimofile, name)
        else:
            self.rimofile = "RIMO_HFI_npipe5v16_symmetrized.fits"
            (
                self.psi_uv,
                self.psi_pol,
                self.fwhm,
                self.ellipticity,
                self.psi_ell,
                self.fsample,
                self.epsilon,
                self.sigma,
            ) = parse_hfi_rimo(self.rimofile, name)
        self._beam = PlanckBeam(name)
        self._bandpass = PlanckBandpass(name)
        self.psi_pol = None
