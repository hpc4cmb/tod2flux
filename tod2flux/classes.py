
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
        integrated_temperature = temperature * self.beam.solid_angle * arcmin ** 2 * 1e-3
        integrated_temperature_err = (
            (temperature_err * beam_solid_angle + temperature * beam.solid_angle_err)
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

    
class Bandpass:
    """ Virtual class that defines the bandpass interface
    """

    @property
    def mkcmb2mjysr(self):
        """ Return the unit conversion coeffient
        """
        raise RuntimeError("Fell through to virtual method for mkcmb2mjysr")


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
