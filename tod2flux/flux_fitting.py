import os
import sys

import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .utilities import to_UTC, to_date, to_JD, to_MJD, to_DJD, DJDtoUNIX


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

    """

    def __init__(self, database, scan_length=30, coord="C", IAU_pol=True):
        self.database = database
        self.scan_length = scan_length
        self.coord = coord
        self.freqs = set()
        self.IAU_pol = IAU_pol

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
        self, scans, mode, pol, pointing, data, error, freqscan, color_corrections
    ):
        """ Sort fit data by frequency
        """

        for scan in scans:
            for fit in scan:
                if mode not in fit.entries:
                    continue
                det = fit.detector
                if color_corrections is not None and det in color_corrections:
                    cc = 1 / color_corrections[det]
                else:
                    cc = 1
                freq = fit.frequency
                psi = self._rotate_pol(fit.theta, fit.phi, fit.psi_pol, fit.coord)
                eta = fit.pol_efficiency
                params = fit.entries[mode]
                if freq not in pointing:
                    pointing[freq] = []
                    data[freq] = []
                    error[freq] = []
                    freqscan[freq] = Scan()
                if pol:
                    pointing[freq].append(
                        [1, eta * np.cos(2 * psi), eta * np.sin(2 * psi)]
                    )
                else:
                    pointing[freq].append([1])
                data[freq].append(params.flux * cc)
                error[freq].append(params.flux_err * cc)
                freqscan[freq].append(fit)
        return

    def _solve_data(
        self, pointing, data, error, freqscan, results=None, noiseweight=False
    ):
        """ Solve for polarized flux in each frequency

        noiseweight=False because it will promote T->P leakage.
        """

        flux = {}
        flux_err = {}
        x, y, z = [], [], []
        for freq in sorted(pointing):
            pointing[freq] = np.vstack(pointing[freq])
            data[freq] = np.array(data[freq])
            scan = freqscan[freq]
            print(
                "Solving for flux {} GHz using {} samples".format(freq, data[freq].size)
            )
            error[freq] = np.array(error[freq])
            pnt2 = []
            if noiseweight:
                for pnt, err in zip(pointing[freq], error[freq]):
                    pnt2.append(pnt / err ** 2)
                    pnt2 = np.vstack(pnt2).T.copy()
            else:
                pnt2 = pointing[freq].T.copy() / np.median(error[freq]) ** 2
            invcov = np.dot(pnt2, pointing[freq])
            rcond = 1 / np.linalg.cond(invcov)
            if rcond > 1e-3:
                # Full polarized solution
                cov = np.linalg.inv(invcov)
                proj = np.dot(pnt2, data[freq])
                flux[freq] = np.dot(cov, proj)
                flux_err[freq] = cov
            else:
                # Intensity only solution
                cov = 1 / invcov[0, 0]
                proj = np.dot(pnt2[0], data[freq])
                flux[freq] = np.array([cov * proj, 0, 0])
                flux_err[freq] = np.diag(np.array([cov, 0, 0]))
            print("freq = {}".format(freq))
            print("  time = {}".format(scan.datetime()))
            print(
                "  flux = {} +- {}".format(flux[freq], np.sqrt(np.diag(flux_err[freq])))
            )
            if results is not None:
                scale = 1e3  # mJy
                I, Q, U = flux[freq] * scale
                Ierr, Qerr, Uerr = np.sqrt(np.diag(flux_err[freq])) * scale
                results.write(
                    "{:12}, {:>26}, {:12.1f}, {:13.1f}, {:12.1f}, {:13.1f}, {:12.1f}, {:13.1f}\n".format(
                        freq, scan.date(), I, Ierr, Q, Qerr, U, Uerr
                    )
                )
            x.append(freq)
            y.append(flux[freq])
            z.append(np.diag(flux_err[freq]) ** 0.5)

        return flux, flux_err, x, y, z

    def _plot_flux(self, axes, pol, iscan, x, y, z, scan):

        # Plot the results

        freq = np.array(x)
        for f in freq:
            self.freqs.add(f)
        try:
            if pol:
                I, Q, U = np.vstack(y).T.copy()
                I_err, Q_err, U_err = np.vstack(z).T.copy() ** 0.5
            else:
                (I,) = np.vstack(y).T.copy()
                (I_err,) = np.vstack(z).T.copy() ** 0.5
        except:
            # import pdb
            # pdb.set_trace()
            return
        # Intensity
        axes[0].set_title("Intensity")
        axes[0].errorbar(
            freq, I, I_err, label="scan {} : {}".format(iscan, scan.date()), fmt="o-",
        )
        # axes[0].set_yscale("log")
        axes[0].set_ylabel("Flux [Jy]")
        if pol:
            good = np.logical_or(Q != 0, U != 0)
            freq = freq[good]
            I = I[good]
            Q = Q[good]
            U = U[good]
            I_err = I_err[good]
            Q_err = Q_err[good]
            U_err = U_err[good]
            # Polarization fraction
            axes[1].set_title("Polarization fraction")
            p = np.sqrt(Q ** 2 + U ** 2) / I
            p_err = 0
            axes[1].errorbar(
                freq,
                p,
                p_err,
                label="scan {} : {}".format(iscan, scan.date()),
                fmt="o-",
            )
            axes[1].set_ylim([0, 0.2])
            # Polarization angle
            axes[2].set_title("Polarization angle, IAU = {}".format(self.IAU_pol))
            angle = np.degrees(0.5 * np.arctan2(U, Q)) % 180
            med_angle = np.median(angle)
            for i, ang in enumerate(angle):
                if np.abs(ang - 180 - med_angle) < np.abs(ang - med_angle):
                    angle[i] -= 180
                elif np.abs(ang + 180 - med_angle) < np.abs(ang - med_angle):
                    angle[i] += 180
            angle_err = 0
            axes[2].errorbar(
                freq,
                angle,
                angle_err,
                label="scan {} : {}".format(iscan, scan.date()),
                fmt="o-",
            )
            axes[2].set_ylim([-30, 210])
            for ang in 0, 180:
                axes[2].axhline(ang, linestyle="--", color="k", zorder=0)
            axes[2].set_yticks(np.arange(7) * 30)
        for ax in axes:
            ax.grid(True)
            ax.set_xscale("log")
            ax.set_xticks(sorted(self.freqs))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.tick_params(
                axis="x", which="minor", bottom=False, top=False, labelbottom=False,
            )
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
        self,
        scans,
        mode,
        pol,
        flux,
        flux_err,
        residuals,
        residual_errors,
        color_corrections,
    ):
        # Measure the residual per detector

        for scan in scans:
            for fit in scan:
                if mode not in fit.entries:
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
                params = fit.entries[mode]
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
        return variable

    def fit(self, target, pol=True, color_corrections=None, fname="results.csv"):
        if color_corrections is None:
            ccstring = ""
        else:
            ccstring = "_ccorrected"
        # Get all entries in the database for this target.
        # Each entry is a list of Fit objects
        all_fits = self.database.targets[target]
        scans = self.find_scans(all_fits)
        mode = "NLFit6"
        # mode = "LinFit4"
        fig = plt.figure(figsize=[18, 12])
        axes = []
        for i in range(3):
            axes.append(fig.add_subplot(2, 2, 1 + i))

        results = open(fname, "w")
        results.write("# Target = {}\n".format(target))
        results.write(
            "# {:>10}, {:>26}, {:>12}, {:>13}, {:>12}, {:>13}, {:>12}, {:>13}\n".format(
                "freq [GHz]",
                "time",
                "I flux [mJy]",
                "I error [mJy]",
                "Q flux [mJy]",
                "Q error [mJy]",
                "U flux [mJy]",
                "U error [mJy]",
            )
        )
        residuals = {}
        residual_errors = {}

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
            pointing = {}
            data = {}
            error = {}
            freqscan = {}

            self._sort_data(
                [scan], mode, pol, pointing, data, error, freqscan, color_corrections
            )

            for freq in pointing:
                if freq not in all_pointing:
                    all_pointing[freq] = []
                    all_data[freq] = []
                    all_error[freq] = []
                    all_freqscan[freq] = Scan()
                all_pointing[freq] += pointing[freq]
                all_data[freq] += data[freq]
                all_error[freq] += error[freq]
                all_freqscan[freq] += freqscan[freq]

            flux, flux_err, x, y, z = self._solve_data(
                pointing, data, error, freqscan, results
            )
            if len(flux) == 0:
                continue
            all_flux.append(flux)
            all_flux_err.append(flux_err)
            self._measure_residual(
                [scan],
                mode,
                pol,
                flux,
                flux_err,
                residuals,
                residual_errors,
                color_corrections,
            )
            self._plot_flux(axes, pol, iscan + 1, x, y, z, scan)

        variable = self._is_variable(all_flux, all_flux_err)

        if not variable:

            flux, flux_err, x, y, z = self._solve_data(
                all_pointing, all_data, all_error, all_freqscan, results
            )
            self._measure_residual(
                scans,
                mode,
                pol,
                flux,
                flux_err,
                residuals,
                residual_errors,
                color_corrections,
            )
            self._plot_flux(axes, pol, "Combined", x, y, z, all_scan)

        fig.suptitle(
            "{} - {}, variable = {}, {}, coord = {}".format(
                target, mode, variable, all_scan.date(), self.coord
            )
        )
        plt.legend(loc="upper left", bbox_to_anchor=[1, 1])
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
