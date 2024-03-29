from collections import OrderedDict
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import degree, arcmin
import scipy.optimize

try:
    import quaternionarray as qa
except ModuleNotFoundError:
    try:
        import toast.qarray as qa
    except:
        print("quaternionarray or toast is required for quaternion manipulation")
        raise

from .classes import FitEntry, Fit
from .kernels import fast_bin_map
from .utilities import to_UTC, to_date, to_JD, to_MJD, to_DJD, DJDtoUNIX


XAXIS, YAXIS, ZAXIS = np.eye(3)


class DetectorFitter:
    """ A class that fits Detector against Dataset

    Input:
        detector(Detector) : an instance of the Detector class
        gap_length_s (float) : minimum separation in time stamps to split
             the scan [seconds]
        fit_radius_arcmin (float) : fitting radius in arc minutes.
        correlation_length (float):  time between statistically independent samples
        njackknife (int):  Number of jackknife subsets to test for uncertainties
    """

    def __init__(
        self,
        detector,
        gap_length_s=86400,
        fit_radius_arcmin=1000,
        correlation_length=60,
        njackknife=25,
        sim_mode=False,
    ):
        self.detector = detector
        self.gap_length_s = gap_length_s
        self.fit_radius_arcmin = fit_radius_arcmin
        self.wbin = max(2, detector.fwhm_arcmin / 5)
        self.radius = detector.fwhm_arcmin * 2
        self.nbin = int(2 * self.radius / self.wbin + 1)
        self.bin_center = np.arange(self.nbin) * self.wbin - self.radius + self.wbin / 2
        self.bin_lim = np.arange(self.nbin + 1) * self.wbin - self.radius
        self.Y, self.X = np.meshgrid(self.bin_center, self.bin_center)
        self.Y_lim, self.X_lim = np.meshgrid(self.bin_lim, self.bin_lim)
        self.correlation_length = correlation_length
        self.njackknife = njackknife
        self.sim_mode = sim_mode

    def fit(self, dataset):
        print(
            "Fitting '{}' against '{}'".format(self.detector.name, dataset.name),
            flush=True,
        )
        # There will be one Fit object for every scan
        self.fits = []

        scans = self.identify_scans(dataset.time_s)
        print("Found {} scans: {}".format(len(scans), scans), flush=True)  # DEBUG

        for iscan, ind in enumerate(scans):
            self.fit_scan(dataset, iscan, ind)

        return self.fits

    def fit_scan(self, dataset, iscan, ind):
        """ Fit the detector against one scan of the target
        """
        target_phi = dataset.target_phi
        target_theta = dataset.target_theta

        times = dataset.time_s[ind]
        print(
            "Scan # {}, wall time = {:.2f} days, detector time = {:.2f} min, "
            "nsample = {}".format(
                iscan,
                (times[-1] - times[0]) / 86400,
                times.size / self.detector.fsample / 60,
                times.size,
            ),
            flush=True,
        )
        theta = dataset.theta[ind]
        phi = dataset.phi[ind]

        outer, inner, quadrants = self.crop_data(target_theta, target_phi, theta, phi)
        print("\nOUTER = {}, INNER = {}\n".format(np.sum(outer), np.sum(inner)))
        ninner = np.sum(inner)
        if ninner < 10:
            print("Empty scan, no processing done", flush=True)
            return None
        if np.any(quadrants < 0.1 * ninner):
            print("Incomplete scan, no processing done", flush=True)
            return None
        times = times[outer]
        theta = theta[outer].copy()
        phi = phi[outer].copy()

        psi = dataset.psi[ind][outer].copy()
        signal_mK = dataset.signal_mK[ind][outer].copy()

        psi_pol = np.median(psi)
        # Remove the polarizer angle from psi
        psi -= np.radians((self.detector.psi_pol_deg + 180))

        pol_eff = self.detector.pol_efficiency
        frequency = self.detector.nominal_frequency

        result = self.rotate_data(theta, phi, psi, target_phi, target_theta)
        if result is None:
            return None
        scan_phi_arcmin, scan_theta_arcmin = result

        if self.sim_mode:
            self.simulate(
                times, scan_phi_arcmin, scan_theta_arcmin, signal_mK, psi_pol, pol_eff,
            )

        fit = Fit(
            dataset.name,
            dataset.target,
            dataset.target_theta,
            dataset.target_phi,
            dataset.coord,
            self.detector.name,
            self.detector.detector_set,
            times,
            psi_pol,
            pol_eff,
            frequency,
        )
        self.fits.append(fit)

        self.plot_signal(
            scan_phi_arcmin,
            scan_theta_arcmin,
            signal_mK,
            new=True,
            nrow=3,
            ncol=3,
            title="Full signal",
            suptitle="{} -- {} -- scan # {}, info = {}, coord = {:.4f} {:.4f} ({}), {} - {}"
            "".format(
                dataset.target,
                self.detector.name,
                iscan,
                dataset.info,
                dataset.target_lon_deg,
                dataset.target_lat_deg,
                dataset.coord,
                to_date(times[0]),
                to_date(times[-1]),
            ),
        )

        self.linear_fit(
            times,
            scan_phi_arcmin,
            scan_theta_arcmin,
            signal_mK,
            fit,
            fit_offset=False,
            fit_gradient=False,
            derivative_order=0,
        )
        self.linear_fit(
            times,
            scan_phi_arcmin,
            scan_theta_arcmin,
            signal_mK,
            fit,
            fit_offset=False,
            fit_gradient=True,
            derivative_order=0,
        )
        # self.linear_fit(
        #    times,
        #    scan_phi_arcmin,
        #    scan_theta_arcmin,
        #    signal_mK,
        #    fit,
        #    fit_offset=False,
        #    fit_gradient=True,
        #    derivative_order=1,
        # )
        self.linear_fit(
            times,
            scan_phi_arcmin,
            scan_theta_arcmin,
            signal_mK,
            fit,
            fit_offset=True,
            fit_gradient=True,
            derivative_order=0,
        )
        self.nonlinear_fit(
            times,
            scan_phi_arcmin,
            scan_theta_arcmin,
            signal_mK,
            fit,
            fit_offset=True,
            fit_gradient=True,
            derivative_order=0,
        )
        # self.nonlinear_fit(
        #    times,
        #    scan_phi_arcmin,
        #    scan_theta_arcmin,
        #    signal_mK,
        #    fit,
        #    fit_offset=True,
        #    fit_gradient=True,
        #    derivative_order=1,
        # )

        self.fig.subplots_adjust(hspace=0.4)
        fname_plot = "plot_{}_{}_{}.png".format(
            dataset.target, self.detector.name, iscan
        )
        plt.savefig(fname_plot)
        plt.close()
        print("Plot saved in", fname_plot, flush=True)

        return self.fits

    def plot_signal(
        self,
        phi,
        theta,
        signal,
        new=False,
        nrow=1,
        ncol=1,
        title="binned",
        model=None,
        entry=None,
        suptitle=None,
    ):
        if new:
            self.fig = plt.figure(figsize=[8 * ncol, 4 * nrow])
            if suptitle is not None:
                self.fig.suptitle(suptitle)
            self.nrow = nrow
            self.ncol = ncol
            self.iplot = 0
        self.iplot += 1
        sigmap, hitmap = fast_bin_map(
            np.ascontiguousarray(phi, dtype="<f8"),
            np.ascontiguousarray(theta, dtype="<f8"),
            np.ascontiguousarray(signal, dtype="<f8"),
            self.wbin,
            self.radius,
        )
        good = hitmap != 0
        best = hitmap >= np.median(hitmap[good])
        offset = np.median(sigmap[best])
        sigmap[good] -= offset
        amp = np.amax(sigmap[best]) * 0.9
        ax = self.fig.add_subplot(self.nrow, self.ncol, self.iplot)

        def decorate(ax):
            ax.set_aspect("equal")
            ax.set_xlim([self.bin_lim[0], self.bin_lim[-1]])
            ax.set_ylim([self.bin_lim[0], self.bin_lim[-1]])
            cb = plt.colorbar(pc, orientation="horizontal", aspect=30, pad=0.2)
            cb.set_label("mK")
            ax.set_xlabel("Cross-scan [arc min]")
            ax.set_ylabel("In-scan [arc min]")

        if new:
            # First panel will show the distribution of hits
            pc = ax.pcolor(
                self.X_lim, self.Y_lim, sigmap, vmin=-amp, vmax=amp, cmap="coolwarm",
            )
            # mask = np.logical_and(
            #    np.logical_and(phi > np.amin(self.X_lim), phi < np.amax(self.X_lim)),
            #    np.logical_and(theta > np.amin(self.Y_lim), theta < np.amax(self.Y_lim))
            # )
            ax.scatter(phi, theta, s=1, color="k")
            ax.set_title("hits")
            decorate(ax)
            self.iplot += 1
            ax = self.fig.add_subplot(self.nrow, self.ncol, self.iplot)

        pc = ax.pcolor(
            self.X_lim, self.Y_lim, sigmap, vmin=-amp, vmax=amp, cmap="coolwarm",
        )
        if model is not None:
            modelmap, _ = fast_bin_map(
                np.ascontiguousarray(phi, dtype="<f8"),
                np.ascontiguousarray(theta, dtype="<f8"),
                np.ascontiguousarray(model, dtype="<f8"),
                self.wbin,
                self.radius,
            )
            modelmap[good] -= np.median(modelmap[best])
            modelmap[good] /= np.amax(modelmap[best])
            ax.contour(self.X, self.Y, modelmap, [0.01, 0.1, 0.5, 0.9], colors="k")
        ax.set_title(title)
        decorate(ax)

        if entry is not None:
            x = 1.1
            y = 1.0

            def add_text(x, y, text):
                ax.text(
                    x,
                    y,
                    text,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                return y - 0.12

            y = add_text(
                x, y, "flux = {:.3f} +- {:.3f} Jy".format(entry.flux, entry.flux_err)
            )
            y = add_text(x, y, "$\chi^2$ = {:.3f}".format(entry.rchisq))
            for name, value in entry.params.items():
                error = entry.errors[name]
                unit = entry.units[name]
                y = add_text(
                    x, y, "{} = {:.3f} +- {:.3f} {}".format(name, value, error, unit)
                )
        return

    def rotate_data(self, theta, phi, psi, target_phi, target_theta):
        """ Rotate the pointing to the detector frame
        """
        print("Rotating vectors ... ", flush=True)
        t1 = time.time()

        scan_n = theta.size
        scan_vec = np.vstack(
            [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
        ).T.copy()

        psi0 = np.median(psi[scan_n // 2 - 100 : scan_n // 2 + 100])
        print("Mean polarization angle is", psi0 / degree, flush=True)
        mean_psi = psi0
        if self.detector.beam.referential == "Dxx":
            mean_psi += self.detector.psi_pol_deg * degree

        rot1 = qa.rotation(ZAXIS, -target_phi)
        rot2 = qa.rotation(YAXIS, -target_theta)
        rot3 = qa.rotation(ZAXIS, -psi0)
        rot = qa.mult(rot3, qa.mult(rot2, rot1)).ravel()

        scan_vec = qa.rotate(rot, scan_vec)

        scan_phi_arcmin = np.arcsin(scan_vec[:, 0]) / arcmin
        scan_theta_arcmin = np.arcsin(scan_vec[:, 1]) / arcmin

        # check if the scan direction is now what we expected

        ind = slice(scan_n // 2 - 4, scan_n // 2 + 6)
        dx = np.median(np.diff(scan_phi_arcmin[ind]))
        dy = np.median(np.diff(scan_theta_arcmin[ind]))
        rot_tot = np.copy(rot)
        if (
            abs(dx) > 1e-2 or dy < 1 or True
        ):  # ALWAYS rotate based on the scan direction
            # print("ERROR: failed to rotate to scan basis")
            psi0 = np.arctan2(dx, dy) + np.pi
            print(
                "dx = {}, dy = {}, psi = {:.3f}".format(dx, dy, psi0 / degree),
                flush=True,
            )
            rot = qa.rotation(ZAXIS, psi0)
            # for ivec, pvec in enumerate(scan_vec):
            #    scan_vec[ivec] = rot(Geom.Vector(pvec))
            scan_vec = qa.rotate(rot, scan_vec)
            x = scan_vec[:, 0]
            y = scan_vec[:, 1]
            z = scan_vec[:, 2]
            scan_phi_arcmin = (
                np.sign(x) * np.arccos(z / np.sqrt(x ** 2 + z ** 2)) / arcmin
            )
            scan_theta_arcmin = (
                np.sign(y) * np.arccos(z / np.sqrt(y ** 2 + z ** 2)) / arcmin
            )
            dx = np.median(np.diff(scan_phi_arcmin[ind]))
            dy = np.median(np.diff(scan_theta_arcmin[ind]))
            print("dx = {}, dy = {}".format(dx, dy))
            rot_tot = np.dot(rot, rot_tot)
        # if abs(dx) > 1e-1 or dy < 1:
        #    print('ERROR: failed to rotate to scan basis')
        #    print('dx = {}, dy = {}'.format(dx, dy))

        theta0 = 0
        phi0 = 0
        psi0 = 0

        t2 = time.time()
        print("Done in {:6.2f} s".format(t2 - t1), flush=True)

        # Test that we have sufficient coverage for fitting

        tol = self.detector.beam.fwhm_arcmin / 10

        if (
            np.amin(scan_phi_arcmin) > -tol
            or np.amax(scan_phi_arcmin) < tol
            or np.amin(scan_theta_arcmin) > -tol
            or np.amax(scan_theta_arcmin) < tol
        ):
            print("Broken scan, skipping", flush=True)
            return None
        return scan_phi_arcmin, scan_theta_arcmin

    def crop_data(self, target_theta, target_phi, theta, phi):
        """ Crop the data to fit the fitting radius
        """
        print("Computing distances ... ")
        t1 = time.time()

        vec0 = np.vstack(
            [
                np.cos(target_phi) * np.sin(target_theta),
                np.sin(target_phi) * np.sin(target_theta),
                np.cos(target_theta),
            ]
        )
        vec = np.vstack(
            [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
        )
        vec = np.transpose(vec)

        cos_lim_outer = np.cos(self.fit_radius_arcmin * arcmin)
        cos_lim_inner = np.cos(self.radius / 2 * arcmin)
        cos_dist = np.sum(vec * vec0.T, 1)
        outer = cos_dist > cos_lim_outer
        inner = cos_dist > cos_lim_inner
        # Count samples in each quadrant
        dphi = phi - target_phi
        dphi[dphi > np.pi] -= 2 * np.pi
        dtheta = theta - target_theta
        quadrants = np.array(
            [
                np.sum(inner * (dphi > 0) * (dtheta > 0)),
                np.sum(inner * (dphi < 0) * (dtheta > 0)),
                np.sum(inner * (dphi < 0) * (dtheta < 0)),
                np.sum(inner * (dphi > 0) * (dtheta < 0)),
            ]
        )

        t2 = time.time()
        print("Done in {:6.2f} s".format(t2 - t1))
        return outer, inner, quadrants

    def simulate(
        self,
        times,  # UNIX time
        phi,  # arc min
        theta,  # arc min
        signal,  # mK
        psi_pol,
        pol_eff,
    ):
        """ Add a synthetic source onto the data

        """
        print(
            "\nSimulating a source", flush=True,
        )
        t1 = time.time()

        flux = 1.0
        phi_offset = 0
        theta_offset = 0

        beam = self.detector.beam.get_beam(phi + phi_offset, theta + theta_offset)

        mK2MJySr, _ = self.detector.temperature_to_flux(1, 0)

        signal[:] += beam * flux / mK2MJySr

        return

    def linear_fit(
        self,
        times,  # UNIX time
        phi,  # arc min
        theta,  # arc min
        signal,  # mK
        fit,
        fit_offset=False,
        fit_gradient=False,
        derivative_order=0,
    ):
        """ Assume a fixed beam and perform a GLS fit of the templates

        """
        print(
            "\nFitting fixed beam, fit_offset = {}, fit_gradient = {}".format(
                fit_offset, fit_gradient
            ),
            flush=True,
        )
        t1 = time.time()

        sigma = self.detector.sigma_KCMB * 1e3

        # Sort the data by distance to the center
        r = np.sqrt(phi ** 2 + theta ** 2)
        ind = np.argsort(r)
        orig = np.arange(r.size)[ind]
        times = times[ind].copy()
        phi = phi[ind].copy()
        theta = theta[ind].copy()
        signal = signal[ind].copy()
        r = r[ind].copy()

        beam = self.detector.beam.get_beam(phi, theta)

        templates = []
        template_names = []
        template_units = []

        templates.append(beam)
        template_names.append("amplitude")
        template_units.append("mK")

        templates.append(np.ones(len(signal)))
        template_names.append("offset")
        template_units.append("mK")

        if derivative_order > 0:
            # Derivatives are fitted at the sample closest to the center
            i0 = np.argmin(phi ** 2 + theta ** 2)
            t0 = times[i0]
            for order in range(1, derivative_order + 1):
                templates.append(beam * ((times - t0) / 86400) ** order)
                template_names.append("derivative{}".format(order))
                template_units.append("mK/day^{}".format(order))

        if fit_offset:
            delta = 1e-3  # in arc min
            dbeam_dphi = (
                self.detector.beam.get_beam(phi + delta, theta) - beam
            ) / delta
            templates.append(dbeam_dphi)
            template_names.append("phi offset")
            template_units.append("arc min")
            dbeam_dtheta = (
                self.detector.beam.get_beam(phi, theta + delta) - beam
            ) / delta
            templates.append(dbeam_dtheta)
            template_names.append("theta offset")
            template_units.append("arc min")

        if fit_gradient:
            templates.append(phi * 1e-3)
            template_names.append("phi grad")
            template_units.append("uK / arc min")
            templates.append(theta * 1e-3)
            template_names.append("theta grad")
            template_units.append("uK / arc min")

        ntemplate = len(templates)
        templates = np.vstack(templates)

        jackknife_amplitudes = []
        subset = (times // self.correlation_length).astype(np.int) % self.njackknife
        for ijack in range(self.njackknife + 1):
            if ijack < self.njackknife:
                good = subset != ijack
                jacktemplates = templates[:, good].copy()
                jacksignal = signal[good].copy()
            else:
                jacktemplates = templates
                jacksignal = signal
            invcov = np.dot(jacktemplates, jacktemplates.T) / sigma ** 2
            cov = np.linalg.inv(invcov)
            proj = np.dot(jacktemplates, jacksignal) / sigma ** 2
            if ijack < self.njackknife:
                jackknife_amplitudes.append(np.dot(cov, proj))
            else:
                template_amplitudes = np.dot(cov, proj)
        jackknife_amplitudes = np.vstack(jackknife_amplitudes)
        jackknife_errors = (
            (self.njackknife - 1)
            / self.njackknife
            * np.sum((jackknife_amplitudes - template_amplitudes) ** 2, 0)
        )
        # template_errors = np.diag(cov) ** .5
        template_errors = jackknife_errors ** 0.5  # More reliable

        resid = np.copy(signal)
        for amplitude, template in zip(template_amplitudes, templates):
            resid -= amplitude * template
        chisq = sum(resid ** 2) / sigma ** 2
        rchisq = chisq / (resid.size - ntemplate)

        params = OrderedDict()
        param_errors = OrderedDict()
        param_units = OrderedDict()
        for name, amplitude, error, unit in zip(
            template_names, template_amplitudes, template_errors, template_units
        ):
            if name == "amplitude":
                flux, flux_err = self.detector.temperature_to_flux(amplitude, error)
            params[name] = amplitude
            param_errors[name] = error
            param_units[name] = unit
            print("{} = {} +- {} {}".format(name, amplitude, error, unit))

        print("Flux density = {} +- {} Jy".format(flux, flux_err), flush=True)

        t2 = time.time()
        print(
            "Linear fit completed in {:.2f} s with chisq = {:.1f}, "
            "reduced chisq = {:.3f}".format(t2 - t1, chisq, rchisq),
            flush=True,
        )

        entry = FitEntry(
            "LinFit{}".format(ntemplate),
            flux,
            flux_err,
            chisq,
            rchisq,
            params,
            param_errors,
            param_units,
        )
        fit.add_entry(entry)

        """
        rlims = []
        ramplitudes = []
        rerrors = []
        for f in np.linspace(0.5, 1, 100):
            rlim = r[-1] * f
            i = np.argwhere(r < rlim).ravel()[-1]
            ind = slice(0, i + 1)
            ttemplates = templates[:, ind].copy()
            invcov = np.dot(ttemplates, ttemplates.T) / sigma ** 2
            cov = np.linalg.inv(invcov)
            proj = np.dot(ttemplates, signal[ind]) / sigma ** 2
            amplitudes = np.dot(cov, proj)
            rlims.append(rlim)
            ramplitudes.append(amplitudes)
            rerrors.append(np.diag(cov) ** .5)
        rlims = np.array(rlims)
        ramplitudes = np.vstack(ramplitudes).T
        rerrors = np.vstack(rerrors).T
        plt.clf(); plt.errorbar(rlims, ramplitudes[0], rerrors[0]); plt.savefig("test.png")
        import pdb
        pdb.set_trace()
        """

        self.plot_signal(
            phi,
            theta,
            resid,
            title="Residual, LinFit{}".format(ntemplate),
            model=signal - resid,
            entry=entry,
        )

        return

    def nonlinear_fit(
        self,
        times,  # UNIX time
        phi,  # arc min
        theta,  # arc min
        signal,  # mK
        fit,
        fit_offset=False,
        fit_gradient=False,
        derivative_order=0,
    ):
        """ A nonlinear fit allows for arbitrary beam offsets and can support generic
        Gaussian fits.

        """
        print(
            "\nPerforming nonlinear fit. fit_offset = {}, fit_gradient = {}, derivative_order = {}".format(
                fit_offset, fit_gradient, derivative_order
            ),
            flush=True,
        )
        t1 = time.time()

        # Derivatives are fitted at the sample closest to the center
        i0 = np.argmin(phi ** 2 + theta ** 2)
        t0 = times[i0]

        sigma = self.detector.sigma_KCMB * 1e3

        def get_resid(param, times, phi, theta, signal):
            amp = param[0]
            offset = param[1]
            i = 2
            if derivative_order > 0:
                deriv = param[i : i + derivative_order]
                dtime = (times - t0) / 86400
                amp = np.zeros(times.size) + amp
                for order in range(1, derivative_order + 1):
                    amp += deriv[order - 1] * dtime ** order
                    i += 1
            if fit_offset:
                phi_offset, theta_offset = param[i : i + 2]
                phi = phi + phi_offset
                theta = theta + theta_offset
                i += 2
            if fit_gradient:
                phi_grad, theta_grad = param[i : i + 2]
                phi_grad = phi * phi_grad * 1e-3
                theta_grad = theta * theta_grad * 1e-3
                i += 2
            else:
                phi_grad, theta_grad = 0, 0
            model = (
                amp * self.detector.beam.get_beam(phi, theta)
                + offset
                + phi_grad
                + theta_grad
            )
            return signal - model

        param_names = []
        units = []

        param_names.append("amplitude")
        units.append("mK")

        param_names.append("offset")
        units.append("mK")

        offset0 = np.median(signal)
        amp0 = np.amax(signal) - offset0
        x0 = [amp0, offset0]

        if derivative_order > 0:
            for order in range(1, derivative_order + 1):
                param_names.append("derivative{}".format(order))
                units.append("mK/day^{}".format(order))
                x0 += [0]

        if fit_offset:
            param_names.append("phi offset")
            units.append("arc min")
            param_names.append("theta offset")
            units.append("arc min")
            x0 += [0, 0]

        if fit_gradient:
            param_names.append("phi grad")
            units.append("uK / arc min")
            param_names.append("theta grad")
            units.append("uK / arc min")
            x0 += [0, 0]

        nparam = len(param_names)

        result = scipy.optimize.least_squares(
            get_resid,
            x0,
            method="lm",
            args=(times, phi, theta, signal),
            max_nfev=100,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )

        if not result.success:
            print("WARNING: leastsq failed with {}".format(result.message), flush=True)
            return

        param_amplitudes = result.x
        resid = result.fun
        chisq = sum(resid ** 2) / sigma ** 2
        rchisq = chisq / (resid.size - nparam)

        jacobian = result.jac
        invcov = np.dot(jacobian.T, jacobian)
        cov = np.linalg.inv(invcov) * np.var(resid)
        errors = np.diag(cov) ** 0.5

        params = OrderedDict()
        param_errors = OrderedDict()
        param_units = OrderedDict()
        for name, amplitude, error, unit in zip(
            param_names, param_amplitudes, errors, units
        ):
            if name == "amplitude":
                flux, flux_err = self.detector.temperature_to_flux(amplitude, error)
            params[name] = amplitude
            param_errors[name] = error
            param_units[name] = unit
            print("{} = {} +- {} {}".format(name, amplitude, error, unit), flush=True)

        print("Flux density = {} +- {} Jy".format(flux, flux_err), flush=True)

        t2 = time.time()
        print(
            "Nonlinear_fit completed after {} evaluations in "
            "{:.2f} s with chisq = {:.1f}, reduced chisq = {:.3f}".format(
                result.nfev, t2 - t1, chisq, rchisq
            ),
            flush=True,
        )

        entry = FitEntry(
            "NLFit{}".format(nparam),
            flux,
            flux_err,
            chisq,
            rchisq,
            params,
            param_errors,
            param_units,
        )
        fit.add_entry(entry)

        self.plot_signal(
            phi,
            theta,
            resid,
            title="Residual, NLFit{}".format(nparam),
            model=signal - resid,
            entry=entry,
        )

        return

    def identify_scans(self, times):
        """ Find the gaps between scans
        """
        breaks = np.argwhere(np.diff(times) > self.gap_length_s).ravel() + 1
        breaks = np.hstack([breaks, times.size])
        scans = []
        istart = 0
        for istop in breaks:
            scans.append(slice(istart, istop))
            istart = istop
        return scans

    def __str__(self):
        result = "DetectorFitter:\n"
        result += "detector = {}".format(self.detector)
        return result
