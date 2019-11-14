import os
import sys

import astropy.io.fits as pf
import numpy as np
import scipy.constants as constants
import scipy.integrate as integrate
import scipy.interpolate

from .. import Bandpass


DATADIR = os.path.join(os.path.dirname(__file__), "data")
RIMOFILE_LFI = os.path.join(DATADIR, "RIMO_LFI.fits")
RIMOFILE_HFI = os.path.join(DATADIR, "RIMO_HFI.fits")
TCMB = 2.725
TOPHATS = {
    "LFI18M": (70.28181, 65.21931, 75.34431),
    "LFI18S": (70.36619, 63.67119, 77.06119),
    "LFI19M": (69.62076, 64.49576, 74.74576),
    "LFI19S": (71.02724, 65.31474, 76.73974),
    "LFI20M": (70.09896, 64.57146, 75.62646),
    "LFI20S": (70.54904, 64.98904, 76.10904),
    "LFI21M": (71.06240, 64.79990, 77.32490),
    "LFI21S": (69.58560, 63.83810, 75.33310),
    "LFI22M": (69.80360, 63.89110, 75.71610),
    "LFI22S": (70.84440, 64.20690, 77.48190),
    "LFI23M": (69.95832, 63.20332, 76.71332),
    "LFI23S": (70.68968, 63.84468, 77.53468),
    "LFI24M": (43.95743, 41.98578, 45.92909),
    "LFI24S": (44.23967, 42.15512, 46.32421),
    "LFI25M": (44.11619, 42.05625, 46.17613),
    "LFI25S": (44.08091, 41.97438, 46.18744),
    "LFI26M": (44.03240, 41.97899, 46.08581),
    "LFI26S": (44.16470, 42.11191, 46.21749),
    "LFI27M": (28.32643, 26.04031, 30.61254),
    "LFI27S": (28.58823, 26.15157, 31.02489),
    "LFI28M": (28.74475, 26.78867, 30.70083),
    "LFI28S": (28.16991, 25.88565, 30.45417),
}


def get_correction(
    det, use_afactors=True, use_tophats=True, corrpoly=None, hfiver="204"
):
    """
    Return the bandpass corrections in a dictionary

    Inputs:
        det : planck detector name
        use_afactors(True) : alternative LFI color corrections
        use_tophats(False) : yet another alternative LFI color corrections.
            Requires use_afactors. 
        corrpoly(None) : If supplied, a dictionary of multiplicative
            polynomials to apply to the color correction function
        slope(None) : Possible spectral index (indices) to compute color
            correction for

    Returns:
        correction['mkcmb2mjysr'] : mK_CMB -> MJy / Sr conversion factor
        correction['kcmb2krj'] : K_CMB 2 K_RJ conversion factor
        correction['cc'] : color correction function for powerlaw spectra
        correction['cfreq'] : effective central frequency
    """
    if "LFI" in det:
        rimofile = RIMOFILE_LFI
        detfilter = det[-3:].upper()
        horn = int(detfilter[0:2])
        if horn < 24:
            if use_afactors:
                cfreq = 70.466941e9
                mkmjy = 1.34401e-1
                kcmb2krj = 0.880966
            else:
                # cfreq = 70e9
                cfreq = 70.466941e9
        elif horn < 27:
            if use_afactors:
                cfreq = 44.120803e9
                mkmjy = 5.68877e-2
                kcmb2krj = 0.951172
            else:
                # cfreq = 44e9
                cfreq = 44.120803e9
        else:
            if use_afactors:
                cfreq = 28.455828e9
                mkmjy = 2.43639e-2
                kcmb2krj = 0.979336
            else:
                # cfreq = 30e9
                cfreq = 28.455828e9

        if use_afactors and not use_tophats:
            x = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
            y = {
                "LFI18": np.array(
                    [
                        0.948,
                        0.961,
                        0.972,
                        0.981,
                        0.988,
                        0.994,
                        0.997,
                        0.998,
                        0.997,
                        0.995,
                        0.990,
                        0.983,
                        0.975,
                    ]
                ),
                "LFI19": np.array(
                    [
                        0.856,
                        0.878,
                        0.899,
                        0.919,
                        0.939,
                        0.957,
                        0.975,
                        0.991,
                        1.006,
                        1.020,
                        1.032,
                        1.043,
                        1.053,
                    ]
                ),
                "LFI20": np.array(
                    [
                        0.889,
                        0.908,
                        0.925,
                        0.941,
                        0.956,
                        0.970,
                        0.983,
                        0.994,
                        1.003,
                        1.011,
                        1.018,
                        1.023,
                        1.027,
                    ]
                ),
                "LFI21": np.array(
                    [
                        0.917,
                        0.933,
                        0.947,
                        0.960,
                        0.971,
                        0.981,
                        0.989,
                        0.996,
                        1.001,
                        1.004,
                        1.006,
                        1.006,
                        1.004,
                    ]
                ),
                "LFI22": np.array(
                    [
                        1.024,
                        1.026,
                        1.027,
                        1.026,
                        1.023,
                        1.018,
                        1.011,
                        1.003,
                        0.993,
                        0.982,
                        0.969,
                        0.955,
                        0.940,
                    ]
                ),
                "LFI23": np.array(
                    [
                        0.985,
                        0.991,
                        0.996,
                        0.999,
                        1.001,
                        1.002,
                        1.002,
                        1.000,
                        0.997,
                        0.993,
                        0.988,
                        0.982,
                        0.975,
                    ]
                ),
                "LFI24": np.array(
                    [
                        0.978,
                        0.984,
                        0.988,
                        0.993,
                        0.996,
                        0.998,
                        0.999,
                        1.000,
                        0.999,
                        0.998,
                        0.996,
                        0.993,
                        0.989,
                    ]
                ),
                "LFI25": np.array(
                    [
                        0.967,
                        0.974,
                        0.980,
                        0.985,
                        0.990,
                        0.994,
                        0.996,
                        0.999,
                        1.000,
                        1.000,
                        1.000,
                        0.999,
                        0.997,
                    ]
                ),
                "LFI26": np.array(
                    [
                        0.957,
                        0.966,
                        0.973,
                        0.980,
                        0.985,
                        0.990,
                        0.995,
                        0.998,
                        1.000,
                        1.001,
                        1.002,
                        1.002,
                        1.000,
                    ]
                ),
                "LFI27": np.array(
                    [
                        0.948,
                        0.959,
                        0.969,
                        0.978,
                        0.985,
                        0.991,
                        0.995,
                        0.998,
                        1.000,
                        1.000,
                        0.998,
                        0.995,
                        0.991,
                    ]
                ),
                "LFI28": np.array(
                    [
                        0.946,
                        0.958,
                        0.968,
                        0.977,
                        0.985,
                        0.991,
                        0.996,
                        0.998,
                        1.000,
                        0.999,
                        0.997,
                        0.993,
                        0.988,
                    ]
                ),
            }[det[:-1]]

            if corrpoly == None:
                p = np.polyfit(x, y, 2)
            else:
                p = np.polyfit(x, y * np.polyval(corrpoly[det], x), 2)

            def cc(index):
                return 1 / np.polyval(p, index)

    else:
        rimofile = RIMOFILE_HFI
        detfilter = det.upper()
        cfreq = np.float(det[0:3]) * 1e9

    hdulist = pf.open(rimofile)

    found = False
    for hdu in hdulist:
        if detfilter in hdu.name:
            found = True
            break

    if not found:
        raise Exception("Bandpass for {} not in {}".format(det, rimo))

    correction = {}
    correction["cfreq"] = cfreq * 1e-9
    if cfreq < 9e10:
        if use_afactors and not use_tophats:
            correction["mkcmb2mjysr"] = mkmjy  # * kcmb2krj
            correction["kcmb2krj"] = kcmb2krj
            correction["cc"] = cc
            return correction
        else:
            freq = hdu.data.field(0) * 1e9  # Already in Hz
    else:
        freq = constants.c * hdu.data.field(0) * 1e2  # from 1/cm to Hz
    if use_tophats and det in TOPHATS:
        th = TOPHATS[det]
        trans = hdu.data.field(1) * 0
        trans[np.logical_and(freq >= th[1] * 1e9, freq <= th[2] * 1e9)] = 1
    else:
        # Assume nothing about the transmission normalization
        trans = hdu.data.field(1)
    ind = trans > 1e-6

    # v3.02 bandpasses have duplicated frequencies ...
    for i in range(len(freq) - 1):
        if freq[i] == freq[i + 1]:
            ind[i + 1] = False

    freq = freq[ind]
    trans = trans[ind]
    # trans /= np.max(trans)
    trans /= scipy.integrate.simps(trans, freq)

    # The calculation is a copy from the Hildebrandt and Macias-Perez IDL module

    nu_cmb = constants.k * TCMB / constants.h
    alpha = 2.0 * constants.k ** 3 * TCMB ** 2 / constants.h ** 2 / constants.c ** 2

    x = freq / nu_cmb
    db_dt = alpha * x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2
    db_dt_rj = 2 * freq ** 2 * constants.k / constants.c ** 2

    correction["mkcmb2mjysr"] = (
        1e17
        * scipy.integrate.simps(db_dt * trans, freq)
        / scipy.integrate.simps(cfreq / freq * trans, freq)
    )
    correction["kcmb2krj"] = scipy.integrate.simps(
        db_dt * trans, freq
    ) / scipy.integrate.simps(db_dt_rj * trans, freq)

    slopes = np.linspace(-10, 10, 300)
    ccs = np.zeros(len(slopes))
    iras_nom = scipy.integrate.simps((cfreq / freq) * trans, freq)
    # iras_nom = integrate.simps((freq / cfreq)**-2 * trans, freq)
    # iras_nom = integrate.simps((cfreq / freq)**2 * trans, freq)
    for ialpha, alpha in enumerate(slopes):
        flux_int = scipy.integrate.simps((freq / cfreq) ** alpha * trans, freq)
        # flux_int = integrate.simps((freq / cfreq)**alpha * (freq / cfreq)**-1 * trans, freq)
        # flux_int = integrate.simps((freq / cfreq)**(alpha-1) * trans, freq)
        ccs[ialpha] = flux_int / iras_nom

    if corrpoly == None:
        correction["cc"] = scipy.interpolate.interp1d(slopes, ccs, kind="quadratic")
    else:
        correction["cc"] = scipy.interpolate.interp1d(
            slopes, ccs / np.polyval(corrpoly[det], slopes), kind="quadratic"
        )

    return correction


class PlanckBandpass(Bandpass):
    def __init__(self, detector_name):
        self.detector_name = detector_name
        self.correction = get_correction(detector_name)
        # correction['mkcmb2mjysr'] : mK_CMB -> MJy / Sr conversion factor
        # correction['kcmb2krj'] : K_CMB 2 K_RJ conversion factor
        # correction['cc'] : color correction function for powerlaw spectra
        # correction['cfreq'] : effective central frequency

    def mkcmb2mjysr(self):
        return self.correction["mkcmb2mjysr"]
