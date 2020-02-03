#!/usr/bin/env python

""" This pipeline combines concurrent observations from multiple
detectors into an estimate of the polarized flux density.
"""

import argparse
import os
import sys
import time

import numpy as np

import tod2flux


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fit fluxes for bandpass and polarization",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--database", default="fluxes.pck", help="Name of the fit database",
    )
    parser.add_argument(
        "--target", required=False, help="Specific target to fit",
    )
    parser.add_argument(
        "--scan-length-days",
        default=30,
        type=np.float,
        help="Maximum length of a scan in days",
    )
    args = parser.parse_args()
    print("All parameters:")
    print(args, flush=True)
    return args


def main():
    t0 = time.time()
    args = parse_arguments()

    # Load the fit database

    database = tod2flux.Database(args.database)

    # Initialize the fitter

    fitter = tod2flux.FluxFitter(database, args.scan_length_days)

    # Process the database

    if args.target is not None:
        if args.target not in database.targets:
            raise RuntimeError(
                "{} does not contain {}".format(args.database, args.target)
            )
        targets = [args.target]
    else:
        targets = database.targets

    for target in targets:
        all_fits = database.targets[target]
        print("target =", target, ", fits =", len(all_fits))
        color_corrections, color_correction_errors = fitter.fit(
            target, pol=True, fname="results_{}.csv".format(target)
        )
        color_corrections2, color_correction_errors2 = fitter.fit(
            target,
            color_corrections=color_corrections,
            pol=True,
            fname="ccorrected_results_{}.csv".format(target),
        )
        """
        for det in color_corrections:
            color_corrections[det] *= color_corrections2[det]
        color_corrections2, color_correction_errors2 = fitter.fit(
            target, color_corrections=color_corrections)
        """

    # Save the database

    database.save()

    print("Pipeline completed in {:.2f} s".format(time.time() - t0), flush=True)

    return


if __name__ == "__main__":
    main()
