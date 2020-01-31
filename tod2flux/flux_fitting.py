import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class FluxFitter:
    """ A class that fits the polarized flux density, SED and bandpass
    mismatch corrections against a set of detector flux densities.

    Input:
        database(Database) : An initialized Database object
        scan_length(float) : Maximum length of a single scan across
            the target in days

    """

    def __init__(self, database, scan_length=30):
        self.database = database
        self.scan_length = scan_length

    def _sort_data(self, scans, mode, pol, pointing, data, error, color_corrections):
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
                psi = fit.psi_pol
                eta = fit.pol_efficiency
                params = fit.entries[mode]
                if freq not in pointing:
                    pointing[freq] = []
                    data[freq] = []
                    error[freq] = []
                if pol:
                    pointing[freq].append(
                        [1, eta * np.cos(2 * psi), eta * np.sin(2 * psi)]
                    )
                else:
                    pointing[freq].append([1])
                data[freq].append(params.flux * cc)
                error[freq].append(params.flux_err * cc)
        return

    def _solve_data(self, pointing, data, error, noiseweight=False):
        """ Solve for polarized flux in each frequencye

        noiseweight=False because it will promote T->P leakage.
        """

        flux = {}
        flux_err = {}
        x, y, z = [], [], []
        for freq in sorted(pointing):
            pointing[freq] = np.vstack(pointing[freq])
            data[freq] = np.array(data[freq])
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
            if rcond < 1e-3:
                continue
            cov = np.linalg.inv(invcov)
            proj = np.dot(pnt2, data[freq])
            flux[freq] = np.dot(cov, proj)
            flux_err[freq] = cov
            print("freq = {}".format(freq))
            print("  data = {}".format(data[freq]))
            print("  flux = {}".format(flux[freq]))
            print("  mean(data) = {}".format(np.mean(data[freq])))
            x.append(freq)
            y.append(flux[freq])
            z.append(np.diag(flux_err[freq]) ** 0.5)

        return flux, flux_err, x, y, z

    def _plot_flux(self, axes, pol, iscan, x, y, z):

        # Plot the results

        freq = np.array(x)
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
            freq, I, I_err, label="scan {}".format(iscan), fmt="o-",
        )
        # axes[0].set_yscale("log")
        axes[0].set_ylabel("Flux [Jy]")
        if pol:
            # Polarization fraction
            axes[1].set_title("Polarization fraction")
            p = np.sqrt(Q ** 2 + U ** 2) / I
            p_err = 0
            axes[1].errorbar(
                freq, p, p_err, label="scan {}".format(iscan), fmt="o-",
            )
            axes[1].set_ylim([0, 0.2])
            # Polarization angle
            axes[2].set_title("Polarization angle")
            angle = np.degrees(0.5 * np.arctan2(U, Q)) % 180
            med_angle = np.median(angle)
            for i, ang in enumerate(angle):
                if np.abs(ang - 180 - med_angle) < np.abs(ang - med_angle):
                    angle[i] -= 180
                elif np.abs(ang + 180 - med_angle) < np.abs(ang - med_angle):
                    angle[i] += 180
            angle_err = 0
            axes[2].errorbar(
                freq, angle, angle_err, label="scan {}".format(iscan), fmt="o-",
            )
            axes[2].set_ylim([-20, 200])
            axes[2].axhline(0, linestyle="--", color="k", zorder=0)
            axes[2].axhline(180, linestyle="--", color="k", zorder=0)
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xticks(freq)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.tick_params(
                axis="x", which="minor", bottom=False, top=False, labelbottom=False,
            )
            ax.set_xlabel("Frequency [GHz]")
        return

    def _analyze_residual(self, target, residuals, residual_errors, ccstring):
        for single in True, False:
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
                    ind = slice(0, n // 2)  # Individual scans residuals
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
                # print("  det = {}, flux = {}, estimate = {}, residual = {}".format(
                #     det, detflux, estimate, detflux / estimate))
                residuals[det].append(detflux / estimate)
                residual_errors[det].append(
                    np.sqrt(
                        (detflux_err / estimate) ** 2
                        + (detflux * estimate_err / estimate ** 2) ** 2
                    )
                )

        return

    def fit(self, target, pol=True, color_corrections=None):
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
        fig.suptitle("{} - {}".format(target, mode))
        axes = []
        for i in range(3):
            axes.append(fig.add_subplot(2, 2, 1 + i))

        residuals = {}
        residual_errors = {}

        all_pointing = {}
        all_data = {}
        all_error = {}

        for iscan, scan in enumerate(scans):
            print("\nISCAN = {}\n".format(iscan))
            pointing = {}
            data = {}
            error = {}

            self._sort_data([scan], mode, pol, pointing, data, error, color_corrections)

            for freq in pointing:
                if freq not in all_pointing:
                    all_pointing[freq] = []
                    all_data[freq] = []
                    all_error[freq] = []
                all_pointing[freq] += pointing[freq]
                all_data[freq] += data[freq]
                all_error[freq] += error[freq]

            flux, flux_err, x, y, z = self._solve_data(pointing, data, error)
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
            self._plot_flux(axes, pol, iscan + 1, x, y, z)

        flux, flux_err, x, y, z = self._solve_data(all_pointing, all_data, all_error)
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
        self._plot_flux(axes, pol, "Combined", x, y, z)

        plt.legend(loc="best")
        fname = "flux_fit_{}{}.png".format(target, ccstring)
        plt.savefig(fname)
        print("Fit saved in {}".format(fname), flush=True)

        new_color_corrections, new_color_correction_errors = self._analyze_residual(
            target, residuals, residual_errors, ccstring,
        )

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
        scan_start = -1e30
        for start, fit in zip(starts, fits):
            if start - scan_start > self.scan_length * 86400:
                # Start a new scan
                scan = []
                scans.append(scan)
                scan_start = start
            scan.append(fit)
        return scans
