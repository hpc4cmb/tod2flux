from collections import OrderedDict
import os
import sys

import numpy as np
from scipy.constants import degree, arcmin


class Detector:
    """ Virtual class that defines the detector interface
    """

    @property
    def beam(self):
        raise RuntimeError("Fell through to virtual method for beam")

    @property
    def bandpass(self):
        raise RuntimeError("Fell through to virtual method for bandpass")

    def temperature_to_flux(self, temperature, temperature_err):
        """Returns the flux based on beam fit amplitude and solid angle
        Units:
        temperature: mK
        frequency: GHz
        beam_solid_angle: arcmin^2
        output: MJy / sr
        """
        integrated_temperature = (
            temperature * self.beam.solid_angle * arcmin ** 2 * 1e-3
        )
        integrated_temperature_err = (
            (
                temperature_err * self.beam.solid_angle
                + temperature * self.beam.solid_angle_err
            )
            * arcmin ** 2
            * 1e-3
        )
        # tcmb = 2.725
        # freq = frequency * 1e9
        # x = h * freq / k / tcmb
        # cmb2ant = (x / (np.exp(x/2)-np.exp(-x/2)))**2 # from K_CMB to K_RJ
        # flux = 2 * freq**2 * k * integrated_temperature * cmb2ant / c**2 * 1e26
        # flux_err = 2 * freq**2 * k * integrated_temperature_err * cmb2ant / c**2 * 1e26
        flux = integrated_temperature * self.bandpass.mkcmb2mjysr * 1e3 * 1e6
        flux_err = integrated_temperature_err * self.bandpass.mkcmb2mjysr * 1e3 * 1e6

        return flux, flux_err

    def __str__(self):
        result = "detector\n"
        result += "  name = {}\n".format(self.name)
        result += "  fsample = {} Hz\n".format(self.fsample)
        result += "  psi_pol = {} deg\n".format(self.psi_pol_deg)
        result += "  pol_efficiency = {}\n".format(self.pol_efficiency)
        result += "  sigma = {} K_CMB\n".format(self.sigma_KCMB)
        result += "  beam = {}\n".format(self.beam)
        result += "  bandpass = {}\n".format(self.bandpass)
        return result

    @property
    def psi_pol_deg(self):
        raise RuntimeError("Fell through to virtual method for psi_pol_deg")

    @property
    def sigma_KCMB(self):
        raise RuntimeError("Fell through to virtual method for sigma_KCMB")

    @property
    def pol_efficiency(self):
        raise RuntimeError("Fell through to virtual method for pol_efficiency")

    @property
    def nominal_frequency(self):
        raise RuntimeError("Fell through to virtual method for nominal_frequency")


class Bandpass:
    """ Virtual class that defines the bandpass interface
    """

    @property
    def mkcmb2mjysr(self):
        """ Return the unit conversion coeffient
        """
        raise RuntimeError("Fell through to virtual method for mkcmb2mjysr")

    def __str__(self):
        result = "<bandpass; "
        result += "mkcmb2mjysr = {}".format(self.mkcmb2mjysr)
        result += ">"
        return result


class Beam:
    """ Virtual class that defines the Beam interface
    """

    @property
    def solid_angle(self):
        """ Return the beam solid angle
        """
        raise RuntimeError("Fell through to virtual method for solid_angle")

    @property
    def solid_angle_err(self):
        """ Return the beam solid angle error
        """
        raise RuntimeError("Fell through to virtual method for solid_angle error")

    def get_beam(self, phi_arcmin, theta_arcmin, **kwargs):
        """ Return the beam evaluated at given offsets
        """
        raise RuntimeError("Fell through to virtual method for get_beam")

    def __str__(self):
        result = "<beam; "
        result += "solid_angle = {}".format(self.solid_angle)
        result += ">"
        return result


class FitEntry:
    """ A class representing a particular fit result
    """

    def __init__(
        self,
        mode,
        flux,
        flux_err,
        chisq,
        rchisq,
        fit_params,
        fit_errors,
        fit_units,
        extra_params=None,
    ):
        self.mode = mode
        self.flux = flux
        self.flux_err = flux_err
        self.chisq = chisq
        self.rchisq = rchisq
        self.nparam = len(fit_params)
        self.params = fit_params
        self.errors = fit_errors
        self.units = fit_units
        self.extra = extra_params


class Fit:
    """ A class that contains the results of fitting one scan and one detector.

    """

    def __init__(
        self,
        dataset,
        target,
        theta,
        phi,
        coord,
        detector,
        times,
        psi_pol_rad,
        pol_efficiency,
        frequency,
    ):
        self.dataset = dataset
        self.target = target
        self.theta = theta
        self.phi = phi
        self.coord = coord
        self.detector = detector
        self.start_time = times[0]
        self.stop_time = times[-1]
        self.psi_pol = psi_pol_rad
        self.pol_efficiency = pol_efficiency
        self.frequency = frequency
        self.nsample = times.size
        self.nentry = 0
        self.entries = OrderedDict()

    def add_entry(self, entry):
        self.nentry += 1
        self.entries[entry.mode] = entry
