import os
import sys

import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from .utilities import to_UTC, to_date, to_JD, to_MJD, to_DJD, DJDtoUNIX


def solve_IQU_and_slope(
    pointing, error, data, relative_freq, noiseweight, flux, flux_err, key
):
    """ Use nonlinear fitting to solve for IQU flux across frequencies
    """

    # First measure the slope

    pointing = pointing.T.copy()

    def get_resid(param, pointing, data, relative_freq):
        I, slope = param
        freqfactor = relative_freq ** slope
        model = freqfactor * pointing[0] * I
        return data - model

    x0 = np.array([np.mean(data), 0])

    result = scipy.optimize.least_squares(
        get_resid,
        x0,
        method="lm",
        args=(pointing, data, relative_freq),
        max_nfev=10000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    if not result.success:
        return False

    I, slope = result.x
    # resid = result.fun
    # jacobian = result.jac
    # invcov = np.dot(jacobian.T, jacobian)
    # cov = np.linalg.inv(invcov) * np.var(resid)

    pointing *= relative_freq ** slope

    success = solve_IQU(
        pointing.T.copy(), error, data, noiseweight, flux, flux_err, key
    )

    return success


def solve_IQU(pointing, error, data, noiseweight, flux, flux_err, key):
    """ Use linear regression to solve for polarized IQU flux
    """
    pnt2 = []
    if noiseweight:
        for pnt, err in zip(pointing, error):
            pnt2.append(pnt / err ** 2)
        pnt2 = np.vstack(pnt2).T.copy()
    else:
        pnt2 = pointing.T.copy() / np.median(error) ** 2
    invcov = np.dot(pnt2, pointing)
    rcond = 1 / np.linalg.cond(invcov)
    if rcond > 1e-3:
        # Full polarized solution
        cov = np.linalg.inv(invcov)
        proj = np.dot(pnt2, data)
        result = np.dot(cov, proj)
        if result[0] <= 0:
            return False
        flux[key] = result
        flux_err[key] = cov
    else:
        # Intensity only solution
        cov = 1 / invcov[0, 0]
        proj = np.dot(pnt2[0], data)
        if proj <= 0:
            return False
        flux[key] = np.array([cov * proj, 0, 0])
        flux_err[key] = np.diag(np.array([cov, 0, 0]))
    return True


class PlotData:
    def __init__(self):
        self.freq = []
        self.I = []
        self.I_low = []
        self.I_high = []
        self.p = []
        self.p_low = []
        self.p_high = []
        self.angle = []
        self.angle_low = []
        self.angle_high = []
        self.pol = []

    def append(
        self, freq, I, I_low, I_high, p, p_low, p_high, angle, angle_low, angle_high
    ):
        self.freq.append(freq)
        self.I.append(I)
        self.I_low.append(I_low)
        self.I_high.append(I_high)
        self.p.append(p)
        self.p_low.append(p_low)
        self.p_high.append(p_high)
        self.angle.append(angle)
        self.angle_low.append(angle_low)
        self.angle_high.append(angle_high)
        self.pol.append(p != 0)

    def process(self):
        self.freq = np.array(self.freq)
        self.I = np.array(self.I)
        self.I_low = np.array(self.I_low)
        self.I_high = np.array(self.I_high)
        self.p = np.array(self.p)[self.pol]
        self.p_low = np.array(self.p_low)[self.pol]
        self.p_high = np.array(self.p_high)[self.pol]
        self.angle = np.degrees(np.array(self.angle)[self.pol])
        self.angle_low = np.degrees(np.array(self.angle_low)[self.pol])
        self.angle_high = np.degrees(np.array(self.angle_high)[self.pol])
        med_angle = np.median(self.angle)
        for i, ang in enumerate(self.angle):
            if np.abs(ang - 180 - med_angle) < np.abs(ang - med_angle):
                self.angle[i] -= 180
                self.angle_low[i] -= 180
                self.angle_high[i] -= 180
            elif np.abs(ang + 180 - med_angle) < np.abs(ang - med_angle):
                self.angle[i] += 180
                self.angle_low[i] += 180
                self.angle_high[i] += 180
        self.pol = np.array(self.pol)
        self.pol_freq = self.freq[self.pol]
        self.I_err = [self.I - self.I_low, self.I_high - self.I]
        self.p_err = [self.p - self.p_low, self.p_high - self.p]
        self.angle_err = [self.angle - self.angle_low, self.angle_high - self.angle]


class Scan:
    """ Utility class for organizing flux fit data into scans
    """

    def __init__(self, start=1e30, stop=-1e30):
        self.start = start
        self.stop = stop
        self.fits = []

    def __iadd__(self, other):
        self.fits += other.fits
        self.start = min(self.start, other.start)
        self.stop = max(self.stop, other.stop)
        return self

    def mjd(self):
        """ Return a string representation of the time span """
        return "MJD {:.2f} - {:.2f}".format(to_MJD(self.start), to_MJD(self.stop))

    def datetime(self):
        """ Return a string representation of the time span """
        return "{} - {}".format(to_UTC(self.start), to_UTC(self.stop))

    def date(self):
        """ Return a string representation of the time span """
        return "{} -- {}".format(to_date(self.start), to_date(self.stop))

    def append(self, fit):
        if self.start > fit.start_time:
            self.start = fit.start_time
        if self.stop < fit.stop_time:
            self.stop = fit.stop_time
        self.fits.append(fit)

    def __getitem__(self, key):
        return self.fits[key]

    def __setitem__(self, key, value):
        self.fits[key] = value

    def __contains__(self, key):
        return key in self.fits


class FluxFitter:
    """ A class that fits the polarized flux density, SED and bandpass
    mismatch corrections against a set of detector flux densities.

    Input:
        database(Database) : An initialized Database object
        scan_length(float) : Maximum length of a single scan across
            the target in days
        coord(str) : Either C, E or G
        IAU_pol(bool) : Use IAU polarization convention instead of
            Healpix convention
        detsets(bool) : Split the fits into detector sets rather than by
            frequency
        net_corrections(dict) : dictionary of correction factors to
             apply to the estimated uncertainties
        mode(string) : Fit mode
        target_dict(dict) :  Dictionary to use to replace target names
             in the database
        do_freqpairs(bool) :  Solve the polarized flux at individual
            frequencies AND using pairs of adjacent frequencies

    """

    def __init__(
        self,
        database,
        scan_length=30,
        coord="C",
        IAU_pol=True,
        detsets=True,
        net_corrections=None,
        mode="LinFit4",
        target_dict=None,
        bgmap=None,
        do_freqpairs=True,
    ):
        self.database = database
        self.scan_length = scan_length
        self.coord = coord
        self.freqs = set()
        self.IAU_pol = IAU_pol
        self.detsets = detsets
        self.net_corrections = net_corrections
        self.mode = mode
        self.target_dict = target_dict
        self.bgmap = bgmap
        if self.bgmap is not None:
            self.sorted_bgmap = np.sort(bgmap)
            npix = self.sorted_bgmap.size
            self.bgmin = self.sorted_bgmap[0]
            self.bgmax = self.sorted_bgmap[int(npix * 0.85)]
        self.do_freqpairs = do_freqpairs
        return

    def _rotate_pol(self, theta_in, phi_in, psi_in, coord_in):
        """ Rotate psi from coord_in to self.coord at (theta_in, phi_in) and
        correct the polarization convention.
        """
        if coord_in.upper() != self.coord.upper():
            dtheta = 1e-6

            vec1_in = hp.dir2vec(theta_in, phi_in)
            vec2_in = hp.dir2vec(theta_in - dtheta, phi_in)

            rotmatrix = hp.rotator.get_coordconv_matrix([coord_in, self.coord])[0]

            vec1_out = np.dot(rotmatrix, vec1_in)
            vec2_out = np.dot(rotmatrix, vec2_in)

            theta_out, phi_out = hp.vec2dir(vec1_out)
            vec3 = hp.dir2vec(theta_out - dtheta, phi_out)
            ang = np.arccos(np.dot(vec2_out - vec1_out, vec3 - vec1_out) / dtheta ** 2)
            psi_out = psi_in + ang
        else:
            psi_out = psi_in

        if self.IAU_pol:
            psi_out *= -1

        return psi_out

    def _sort_data(
        self, scans, pol, frequency, pointing, data, error, freqscan, color_corrections,
    ):
        """ Sort fit data by frequency
        """

        for scan in scans:
            for fit in scan:
                if self.mode not in fit.entries:
                    continue
                det = fit.detector
                if color_corrections is not None and det in color_corrections:
                    cc = 1 / color_corrections[det]
                else:
                    cc = 1
                freq = fit.frequency
                detset = fit.detector_set
                if self.detsets:
                    key = detset
                else:
                    key = freq
                psi = self._rotate_pol(fit.theta, fit.phi, fit.psi_pol, fit.coord)
                eta = fit.pol_efficiency
                params = fit.entries[self.mode]
                if key not in pointing:
                    frequency[key] = []
                    pointing[key] = []
                    data[key] = []
                    error[key] = []
                    freqscan[key] = Scan()
                frequency[key].append(freq)
                if pol:
                    pointing[key].append(
                        [1, eta * np.cos(2 * psi), eta * np.sin(2 * psi)]
                    )
                else:
                    pointing[key].append([1])
                data[key].append(params.flux * cc)
                if self.net_corrections is not None:
                    error[key].append(params.flux_err * cc * self.net_corrections[det])
                else:
                    error[key].append(params.flux_err * cc)
                # error[key][-1] = error[key][-1] ** .5  # TEMPORARY
                freqscan[key].append(fit)
        return

    def _solve_data(
        self,
        all_frequency,
        all_pointing,
        all_data,
        all_error,
        freqscan,
        results=None,
        noiseweight=True,
    ):
        """ Solve for polarized flux in each frequency

        noiseweight=False because it will promote T->P leakage.
        """

        flux = {}
        flux_err = {}
        plot_data = PlotData()
        last_freq = None
        last_pointing = None
        last_data = None
        last_error = None
        for freq_key in sorted(all_pointing):
            for do_pair in True, False:
                if do_pair:
                    freq = np.mean(all_frequency[freq_key])
                    self.freqs.add(int(freq))
                    all_pointing[freq_key] = np.vstack(all_pointing[freq_key])
                    all_data[freq_key] = np.array(all_data[freq_key])
                    all_error[freq_key] = np.array(all_error[freq_key])
                    scan = freqscan[freq_key]

                    pointing = all_pointing[freq_key]
                    data = all_data[freq_key]
                    error = all_error[freq_key]
                    relative_freq = None

                    last_last_freq = last_freq
                    last_last_pointing = last_pointing
                    last_last_data = last_data
                    last_last_error = last_error

                    last_freq = freq
                    last_pointing = pointing
                    last_data = data
                    last_error = error

                    if not self.do_freqpairs or last_last_data is None:
                        continue

                    if noiseweight:
                        freq = (
                            np.sum(last_last_freq / last_last_error ** 2)
                            + np.sum(last_freq / last_error ** 2)
                        ) / (
                            np.sum(1 / last_last_error ** 2)
                            + np.sum(1 / last_error ** 2)
                        )
                    else:
                        freq = 0.5 * (last_last_freq + last_freq)
                    key = "{}+{}".format(int(last_last_freq), int(last_freq))
                    #key = int(freq)
                    pointing = np.vstack([last_last_pointing, last_pointing])
                    data = np.hstack([last_last_data, last_data])
                    error = np.hstack([last_last_error, last_error])
                    relative_freq = (
                        np.hstack(
                            [
                                np.ones(last_last_data.size) * last_last_freq,
                                np.ones(last_data.size) * last_freq,
                            ]
                        )
                        / freq
                    )
                else:
                    key = str(freq_key)
                    pointing = last_pointing
                    data = last_data
                    error = last_error
                    freq = last_freq
                    relative_freq = None

                print(
                    "Solving for flux with {} at {} GHz using {} samples".format(
                        key, freq, data.size
                    )
                )
                if do_pair:
                    success = solve_IQU_and_slope(
                        pointing,
                        error,
                        data,
                        relative_freq,
                        noiseweight,
                        flux,
                        flux_err,
                        key,
                    )
                else:
                    success = solve_IQU(
                        pointing, error, data, noiseweight, flux, flux_err, key
                    )
                if not success:
                    continue
                print("{} {}GHz".format(key, freq))
                print("  time = {}".format(scan.datetime()))
                print(
                    "  flux = {} +- {}".format(
                        flux[key], np.sqrt(np.diag(flux_err[key]))
                    )
                )
                if results is not None:
                    # scale = 1e3  # mJy
                    I, Q, U = flux[key][:3]  # * scale
                    I_err, Q_err, U_err = np.sqrt(np.diag(flux_err[key][:3]))  # * scale
                    (
                        _,
                        I_low,
                        I_high,
                        p,
                        p_low,
                        p_high,
                        angle,
                        angle_low,
                        angle_high,
                    ) = self._derive_pol_bayes(flux[key][:3], flux_err[key][:3])
                    # p, p_low, p_high, angle, angle_err = self._derive_pol(
                    #    I, I_err, Q, Q_err, U, U_err
                    # )
                    scale = 1e3  # mJy
                    results.write(
                        "{:>12}, {:12.3f}, {:>26}, "
                        "{:12.1f}, {:13.1f}, {:12.1f}, {:13.1f}, {:12.1f}, {:13.1f}, "
                        "{:12.4f}, {:12.4f}, {:12.4f}, {:13.3f}, {:10.3f}, {:10.3f}\n".format(
                            key,
                            freq,
                            scan.date(),
                            I * scale,
                            I_err * scale,
                            Q * scale,
                            Q_err * scale,
                            U * scale,
                            U_err * scale,
                            p,
                            p_low,
                            p_high,
                            np.degrees(angle),
                            np.degrees(angle_low),
                            np.degrees(angle_high),
                        )
                    )
                plot_data.append(
                    freq,
                    I,
                    I_low,
                    I_high,
                    p,
                    p_low,
                    p_high,
                    angle,
                    angle_low,
                    angle_high,
                )

        return flux, flux_err, plot_data

    def _derive_pol_bayes(self, flux, flux_err, quantile=0.68):
        I, Q, U = flux
        Ierr, Qerr, Uerr = np.sqrt(np.diag(flux_err))
        if Q == 0 and U == 0:
            # Unpolarized solution
            return (
                I,
                I - Ierr,
                I + Ierr,
                0,
                0,
                0,
                0,
                0,
                0,
            )
        p = np.sqrt(Q ** 2 + U ** 2) / I
        angle = 0.5 * np.arctan2(U, Q)
        if angle < 0:
            angle += np.pi
        invcov = np.linalg.inv(flux_err)
        # scale the covariance to sensible units
        # scale = 1 / np.mean(np.diag(invcov))
        # invcov *= scale

        def likelihood(I0, p0, angle0):
            d = np.array(
                [
                    I - I0,
                    p * I * np.cos(2 * angle) - p0 * I0 * np.cos(2 * angle0),
                    p * I * np.sin(2 * angle) - p0 * I0 * np.sin(2 * angle0),
                ]
            )
            dcov = np.array(
                [
                    invcov[0, 0] * d[0] + invcov[0, 1] * d[1] + invcov[0, 2] * d[2],
                    invcov[1, 0] * d[0] + invcov[1, 1] * d[1] + invcov[1, 2] * d[2],
                    invcov[2, 0] * d[0] + invcov[2, 1] * d[1] + invcov[2, 2] * d[2],
                ]
            )
            dd = d[0] * dcov[0] + d[1] * dcov[1] + d[2] * dcov[2]
            return np.exp(-0.5 * dd)

        n = 100000
        nsigma = 4

        Imin = max(0, I - nsigma * Ierr)
        Imax = I + nsigma * Ierr
        Ivec = np.random.rand(n) * (Imax - Imin) + Imin

        perr = np.sqrt(Qerr ** 2 + Uerr ** 2) / I
        pmin = max(0, p - nsigma * perr)
        pmax = p + nsigma * perr
        pvec = np.random.rand(n) * (pmax - pmin) + pmin

        angleerr = 0.5 / (Q ** 2 + U ** 2) * (np.abs(Q) * Uerr + np.abs(U) * Qerr)
        anglemin = angle - nsigma * angleerr
        anglemax = angle + nsigma * angleerr
        anglevec = np.random.rand(n) * (anglemax - anglemin) + anglemin

        prob = likelihood(Ivec, pvec, anglevec)
        ind = np.argsort(prob)[::-1]
        sorted_prob = prob[ind]
        total_prob = np.sum(prob)

        i1sigma = 0
        cumulative_prob = 0
        while cumulative_prob < 0.68 * total_prob:
            cumulative_prob += sorted_prob[i1sigma]
            i1sigma += 1
        Icut1 = Ivec[ind][:i1sigma]
        pcut1 = pvec[ind][:i1sigma]
        anglecut1 = anglevec[ind][:i1sigma]

        i2sigma = 0
        cumulative_prob = 0
        while cumulative_prob < 0.95 * total_prob:
            cumulative_prob += sorted_prob[i2sigma]
            i2sigma += 1
        Icut2 = Ivec[ind][:i2sigma]
        pcut2 = pvec[ind][:i2sigma]
        anglecut2 = anglevec[ind][:i2sigma]

        Imin = np.amin(Icut1)
        Imax = np.amax(Icut1)
        if np.amin(pcut2) < 0.01:
            # No lower limit on polarization fraction, use two sigma upper limit
            pmin = 0
            pmax = np.amax(pcut2)
        else:
            pmin = np.amin(pcut1)
            pmax = np.amax(pcut1)
        anglemin = np.amin(anglecut1)
        anglemax = np.amax(anglecut1)

        # Replace the polarization fraction estimate using the asymptotic estimator

        if np.abs(Qerr ** 2 - Uerr ** 2) < 1e-6:
            noise_bias = Qerr * Uerr / I ** 2
        else:
            rho = flux_err[1, 2] / np.sqrt(flux_err[1, 1] * flux_err[2, 2])
            theta = 0.5 * np.arctan2(2 * rho * Qerr * Uerr, Qerr ** 2 - Uerr ** 2)
            sigma_Qsquared = (
                (Qerr * np.cos(theta)) ** 2
                + (Uerr * np.sin(theta)) ** 2
                + rho * Qerr * Uerr * np.sin(2 * theta)
            )
            sigma_Usquared = (
                (Qerr * np.sin(theta)) ** 2
                + (Uerr * np.cos(theta)) ** 2
                - rho * Qerr * Uerr * np.sin(2 * theta)
            )
            noise_bias = (
                sigma_Usquared * np.cos(2 * angle - theta) ** 2
                + sigma_Qsquared * np.sin(2 * angle - theta) ** 2
            ) / I ** 2

        if p ** 2 > noise_bias and np.amax(pcut1) < 0.4:
            p = np.sqrt(p ** 2 - noise_bias)
            pmin = np.amin(pcut1)
            pmax = np.amax(pcut1)
        else:
            p = 0
            pmin = 0
            pmax = np.amax(pcut2)

        return (I, Imin, Imax, p, pmin, pmax, angle, anglemin, anglemax)

    def _derive_pol(self, I, I_err, Q, Q_err, U, U_err):
        """ Calculate the polarization fraction and polarization angle

        The errors are asymmetric, we use Monte Carlo to derive confidence limits
        """
        I = np.atleast_1d(I)
        I_err = np.atleast_1d(I_err)
        Q = np.atleast_1d(Q)
        Q_err = np.atleast_1d(Q_err)
        U = np.atleast_1d(U)
        U_err = np.atleast_1d(U_err)
        p = np.sqrt(Q ** 2 + U ** 2) / I
        angle = np.degrees(0.5 * np.arctan2(U, Q)) % 180
        nmc = 1000
        n = I.size
        # Run a Monte Carlo to measure the bias on polarization fraction
        p_sim0 = np.zeros([nmc, n])
        # angle_sim0 = np.zeros([nmc, n])
        for mc in range(nmc):
            I_sim = I + np.random.randn(I.size) * I_err
            Q_sim = np.random.randn(Q.size) * Q_err
            U_sim = np.random.randn(U.size) * U_err
            p_sim0[mc] = np.sqrt(Q_sim ** 2 + U_sim ** 2) / I_sim
            # temp = np.degrees(0.5 * np.arctan2(U_sim, Q_sim)) % 180
            # Choose the right branch
            # temp[np.abs(temp + 180 - angle) < np.abs(temp - angle)] += 180
            # temp[np.abs(temp - 180 - angle) < np.abs(temp - angle)] -= 180
            # angle_sim[mc] = temp
        p_bias = np.mean(p_sim0, 0)
        p -= p_bias
        # Run a Monte Carlo for the confidence limits
        p_sim = np.zeros([nmc, n])
        angle_sim = np.zeros([nmc, n])
        for mc in range(nmc):
            I_sim = I + np.random.randn(I.size) * I_err
            Q_sim = Q + np.random.randn(Q.size) * Q_err
            U_sim = U + np.random.randn(U.size) * U_err
            p_sim[mc] = np.sqrt(Q_sim ** 2 + U_sim ** 2) / I_sim - p_bias
            temp = np.degrees(0.5 * np.arctan2(U_sim, Q_sim)) % 180
            # Choose the right branch
            temp[np.abs(temp + 180 - angle) < np.abs(temp - angle)] += 180
            temp[np.abs(temp - 180 - angle) < np.abs(temp - angle)] -= 180
            angle_sim[mc] = temp
        # FIXME: this is where we would get the confidence limits rather than sdev
        p_sim = np.sort(p_sim, 0)
        p_low = p_sim[int(nmc * 0.16)]
        p_high = p_sim[int(nmc * 0.84)]
        # p_err = np.std(p_sim - p, 0)
        angle_err = np.std(angle_sim - angle, 0)
        return p, p_low, p_high, angle, angle_err

    def _average_by_frequency(self, freq, value):
        """ Identify data points in `value` that share frequency and collapse them.
        """
        new_freq = []
        new_value = []
        for f in freq:
            if f in new_freq:
                continue
            ind = freq == f
            new_freq.append(f)
            new_value.append(np.mean(value[ind]))
        return new_freq, new_value

    def _plot_flux(self, axes, pol, iscan, plot_data, scan, target):

        # Plot the results

        plot_data.process()
        if iscan != "Combined":
            plot_freq = (
                plot_data.freq * 1.02 ** iscan
            )  # Stagger data points for readability
            colors = [None] + [
                "tab:blue",
                "tab:orange",
                "tab:green",
                "tab:red",
                "tab:purple",
                "tab:brown",
                "tab:pink",
                "tab:gray",
                "tab:olive",
                "tab:cyan",
            ] * 3
            color = colors[iscan]
        else:
            plot_freq = plot_data.freq
            color = "black"
        # for f in plot_data.freq:
        #    self.freqs.add(f)
        # Intensity
        axes[0].set_title("Intensity")
        axes[0].errorbar(
            plot_freq,
            plot_data.I,
            plot_data.I_err,
            label="scan {} : {}".format(iscan, scan.date()),
            fmt="o",
            color=color,
        )
        xx, yy = self._average_by_frequency(plot_freq, plot_data.I)
        axes[0].plot(xx, yy, color=color)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Flux [Jy]")
        ymin, ymax = axes[0].get_ylim()
        # if ymin < 1e-1:
        axes[0].set_ylim(bottom=5e-2)
        datamax = np.amax(plot_data.I + plot_data.I_err)
        if max(ymax, datamax * 1.2) < 1.1e1:
            axes[0].set_ylim(top=1e1)
        elif max(ymax, datamax * 1.2) < 1.1e2:
            axes[0].set_ylim(top=1e2)
        else:
            axes[0].set_ylim(top=1e3)
        if pol:
            plot_freq = plot_freq[plot_data.pol]
            # Polarization fraction
            axes[1].set_title("Polarization fraction")
            axes[1].errorbar(
                plot_freq,
                plot_data.p,
                plot_data.p_err,
                label="scan {} : {}".format(iscan, scan.date()),
                fmt="o",
                color=color,
            )
            xx, yy = self._average_by_frequency(plot_freq, plot_data.p)
            axes[1].plot(xx, yy, color=color)
            if target not in ["M1"]:
                axes[1].set_ylim([-0.01, 0.4])
                axes[1].axhline(0, color="k")
                # axes[1].set_yscale("log")
            axes[1].set_ylabel("Polarization fraction")
            # Polarization angle
            axes[2].set_title("Polarization angle, IAU = {}".format(self.IAU_pol))
            axes[2].errorbar(
                plot_freq,
                plot_data.angle,
                plot_data.angle_err,
                label="scan {} : {}".format(iscan, scan.date()),
                fmt="o",
                color=color,
            )
            xx, yy = self._average_by_frequency(plot_freq, plot_data.angle)
            axes[2].plot(xx, yy, color=color)
            if target not in ["M1"]:
                axes[2].set_ylim([-30, 210])
                for ang in 0, 180:
                    axes[2].axhline(ang, linestyle="--", color="k", zorder=0)
                axes[2].set_yticks(np.arange(7) * 30)
            axes[2].set_ylabel("Angle [deg]")
        for ax in axes:
            ax.grid(True)
            ax.set_xscale("log")
            # if self.do_freqpairs:
            #    ax.set_xticks(np.array(sorted(self.freqs))[::2])
            # else:
            ax.set_xticks(sorted(self.freqs))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.tick_params(
                axis="x", which="minor", bottom=False, top=False, labelbottom=False,
            )
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
            ax.set_xlabel("Frequency [GHz]")
        return

    def _analyze_residual(self, target, residuals, residual_errors, ccstring, variable):
        if variable:
            single = True
        else:
            single = False

        if single:
            ssingle = "single"
        else:
            ssingle = "combined"
        fig2 = plt.figure(figsize=[18, 12])
        fig2.suptitle("{}, {}".format(target, ssingle))
        ax = fig2.add_subplot(1, 1, 1)
        color_corrections = {}
        color_correction_errors = {}
        ticks = []
        ticklabels = []
        for idet, det in enumerate(sorted(residuals.keys())):
            n = len(residuals[det])
            if single:
                if variable:
                    # Individual scans residuals (combined were never made)
                    ind = slice(0, n)
                else:
                    # Individual scans residuals
                    ind = slice(0, n // 2)
            else:
                ind = slice(n // 2, n)  # The combined scan residuals
            resid = np.array(residuals[det])[ind]
            resid_err = np.array(residual_errors[det])[ind]
            # if not single and ccstring == "_ccorrected":
            #    import pdb
            #    pdb.set_trace()
            color_corrections[det] = np.mean(resid)
            n = resid.size
            color_correction_errors[det] = np.sqrt(np.sum((resid_err / n) ** 2))
            ax.errorbar(np.ones(n) * idet, resid, resid_err, fmt="b.", ms=3)
            ax.errorbar(
                idet,
                color_corrections[det],
                color_correction_errors[det],
                fmt="ro",
                ms=4,
            )
            ticks.append(idet)
            ticklabels.append(det)
        plt.xticks(
            ticks,
            ticklabels,
            rotation="vertical",
            horizontalalignment="center",
            verticalalignment="top",
        )
        ax.grid(True)
        ax.set_ylim([0.8, 1.2])
        fname = "color_correction_{}_{}{}.png".format(target, ssingle, ccstring)
        plt.savefig(fname)
        plt.close()
        print("Color correction saved in {}".format(fname))

        return color_corrections, color_correction_errors

    def _measure_residual(
        self, scans, pol, flux, flux_err, residuals, residual_errors, color_corrections,
    ):
        # Measure the residual per detector

        for scan in scans:
            for fit in scan:
                if self.mode not in fit.entries:
                    continue
                freq = fit.frequency
                if freq not in flux:
                    # Could not combine fluxes
                    continue
                det = fit.detector
                if color_corrections is not None and det in color_corrections:
                    cc = 1 / color_corrections[det]
                else:
                    cc = 1
                freqflux = flux[freq]
                freqflux_err = np.sqrt(np.diag(flux_err[freq]))
                if fit.coord.upper() != self.coord.upper():
                    psi = self._rotate_pol(fit.theta, fit.phi, fit.psi_pol, fit.coord)
                else:
                    psi = fit.psi_pol
                eta = fit.pol_efficiency
                params = fit.entries[self.mode]
                if pol:
                    estimate = (
                        freqflux[0]
                        + eta * freqflux[1] * np.cos(2 * psi)
                        + eta * freqflux[2] * np.sin(2 * psi)
                    )
                    estimate_err = (
                        freqflux_err[0]
                        + eta * freqflux_err[1] * np.cos(2 * psi)
                        + eta * freqflux_err[2] * np.sin(2 * psi)
                    )
                else:
                    estimate = freqflux[0]
                    estimate_err = freqflux_err[0]
                if det not in residuals:
                    residuals[det] = []
                    residual_errors[det] = []
                detflux = params.flux * cc
                if self.net_corrections is not None:
                    detflux_err = params.flux_err * cc * self.net_corrections[det]
                else:
                    detflux_err = params.flux_err * cc
                residuals[det].append(detflux / estimate)
                residual_errors[det].append(
                    np.sqrt(
                        (detflux_err / estimate) ** 2
                        + (detflux * estimate_err / estimate ** 2) ** 2
                    )
                )

        return

    def _is_variable(self, all_flux, all_flux_err):
        """ Examine the flux fits and determine if the source is variable or not.
        """
        if len(all_flux) == 0:
            return False
        freqs = set()
        for flux in all_flux:
            for freq in flux:
                freqs.add(freq)
        variable = False
        for freq in freqs:
            freqflux = []
            freqflux_err = []
            for flux, flux_err in zip(all_flux, all_flux_err):
                if freq in flux:
                    freqflux.append(flux[freq][0])
                    freqflux_err.append(np.sqrt(flux_err[freq][0, 0]))
            freqflux = np.array(freqflux)
            freqflux_err = np.array(freqflux_err)
            good = freqflux_err < 0.1 * freqflux
            mean_flux = np.mean(freqflux[good])
            variable = np.any(np.abs(freqflux[good] - mean_flux) > 0.1 * mean_flux)
            if variable:
                break
            mean_error = np.mean(freqflux_err[good])
            variable = np.any(np.abs(freqflux[good] - mean_flux) > 10 * mean_error)
            if variable:
                break
        return variable

    def fit(
        self, target, pol=True, color_corrections=None, fname="results.csv",
    ):
        if color_corrections is None:
            ccstring = ""
        else:
            ccstring = "_ccorrected"
        if self.target_dict is not None and target in self.target_dict:
            name = target + " a.k.a. " + self.target_dict[target]
        else:
            name = target
        # Get all entries in the database for this target.
        # Each entry is a list of Fit objects
        all_fits = self.database.targets[target]
        scans = self.find_scans(all_fits)
        fig = plt.figure(figsize=[18, 12])
        axes = []
        for i in range(3):
            axes.append(fig.add_subplot(2, 2, 1 + i))

        results = open(fname, "w")
        results.write("# Target = {}\n".format(name))
        results.write(
            "# {:12}, {:>10}, {:>26}, "
            "{:>12}, {:>13}, {:>12}, {:>13}, {:>12}, {:>13},"
            "{:>12}, {:>13}, {:>12}, {:>13}, {:>12}, {:>13}\n".format(
                "band(s)",
                "freq [GHz]",
                "time",
                "I flux [mJy]",
                "I error [mJy]",
                "Q flux [mJy]",
                "Q error [mJy]",
                "U flux [mJy]",
                "U error [mJy]",
                "Pol.Frac",
                "PF low lim",
                "PF high lim",
                "Pol.Ang [deg]",
                "PA low lim",
                "PA high lim",
            )
        )
        residuals = {}
        residual_errors = {}

        all_frequency = {}
        all_pointing = {}
        all_data = {}
        all_error = {}
        all_freqscan = {}

        all_flux = []
        all_flux_err = []
        all_scan = Scan()

        for iscan, scan in enumerate(scans):
            all_scan += scan
            print("\nISCAN = {}, {}\n".format(iscan, scan.datetime()))
            frequency = {}
            pointing = {}
            data = {}
            error = {}
            freqscan = {}

            self._sort_data(
                [scan],
                pol,
                frequency,
                pointing,
                data,
                error,
                freqscan,
                color_corrections,
            )

            for key in pointing:
                if key not in all_pointing:
                    all_frequency[key] = []
                    all_pointing[key] = []
                    all_data[key] = []
                    all_error[key] = []
                    all_freqscan[key] = Scan()
                all_frequency[key] += frequency[key]
                all_pointing[key] += pointing[key]
                all_data[key] += data[key]
                all_error[key] += error[key]
                all_freqscan[key] += freqscan[key]

            flux, flux_err, plot_data = self._solve_data(
                frequency, pointing, data, error, freqscan, results,
            )
            if len(flux) == 0:
                continue
            all_flux.append(flux)
            all_flux_err.append(flux_err)
            self._measure_residual(
                [scan],
                pol,
                flux,
                flux_err,
                residuals,
                residual_errors,
                color_corrections,
            )
            self._plot_flux(axes, pol, iscan + 1, plot_data, scan, target)

        variable = self._is_variable(all_flux, all_flux_err)

        if not variable:

            flux, flux_err, plot_data = self._solve_data(
                all_frequency, all_pointing, all_data, all_error, all_freqscan, results
            )
            self._measure_residual(
                scans,
                pol,
                flux,
                flux_err,
                residuals,
                residual_errors,
                color_corrections,
            )
            self._plot_flux(axes, pol, "Combined", plot_data, all_scan, target)

        title = fig.suptitle(
            "{} - {}, variable = {}, {}, coord = {}".format(
                name, self.mode, variable, all_scan.date(), self.coord
            )
        )
        plt.legend(loc="upper left", bbox_to_anchor=[1, 1])

        theta = all_fits[0][0].theta
        phi = all_fits[0][0].phi
        coord = all_fits[0][0].coord
        # hp.mollview(np.zeros(12) + hp.UNSEEN, coord="G", sub=[2, 3, 6], cbar=False)
        if self.bgmap is None:
            bgmap = np.zeros(12) + hp.UNSEEN
        else:
            bgmap = self.bgmap.copy()
            vec = hp.ang2vec(theta, phi)
            radius1 = np.radians(1)
            radius2 = np.radians(2)
            nside = hp.get_nside(bgmap)
            pix1 = hp.query_disc(nside, vec, radius1)
            pix2 = hp.query_disc(nside, vec, radius2)
            pix = np.array(list(set(pix2) - set(pix1)))
            val = np.median(bgmap[pix])
            bgmap[bgmap < self.bgmin] = self.bgmin
            bgmap[bgmap > self.bgmax] = self.bgmax
            frac = np.argmin(np.abs(self.sorted_bgmap - val)) / bgmap.size
            title.set_text(title.get_text() + " background level = {:.2f}".format(frac))
        hp.mollview(
            bgmap, cmap="magma", title="Position (Galactic)", sub=[2, 3, 6], cbar=False,
        )
        hp.graticule(22.5, verbose=False, color="w")
        hp.projplot(theta, phi, "ro", coord=coord)

        fname = "flux_fit_{}{}.png".format(target, ccstring)
        plt.savefig(fname)
        plt.close()
        print("Fit saved in {}".format(fname), flush=True)

        new_color_corrections, new_color_correction_errors = self._analyze_residual(
            target, residuals, residual_errors, ccstring, variable
        )

        results.close()

        return new_color_corrections, new_color_correction_errors

    def find_scans(self, all_fits):
        """ Returns a list of scans.

        Each entry on the scan list is a list of Fit objects where the
        start times are separated by no more than self.scan_length.
        """

        flat_starts = []
        flat_fits = []
        for fits in all_fits:
            for fit in fits:
                flat_starts.append(fit.start_time)
                flat_fits.append(fit)
        flat_starts = np.array(flat_starts)
        flat_fits = np.array(flat_fits)

        # Sort the fits according to start times
        ind = np.argsort(flat_starts)
        starts = flat_starts[ind]
        fits = flat_fits[ind]

        # Now organize the fits into scans
        scans = []
        scan = None
        for start, fit in zip(starts, fits):
            if scan is None or start - scan.start > self.scan_length * 86400:
                # Start a new scan
                scan = Scan()
                scans.append(scan)
            scan.append(fit)
        return scans
