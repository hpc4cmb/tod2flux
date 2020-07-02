#!/usr/bin/env python

""" This pipeline combines concurrent observations from multiple
detectors into an estimate of the polarized flux density.
"""

import argparse
import os
import pickle
import sys
import time

import healpy as hp
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
        "--coord", default="C", help="Reference coordinate system [G,E,C]",
    )
    parser.add_argument(
        "--scan-length-days",
        default=30,
        type=np.float,
        help="Maximum length of a scan in days",
    )
    parser.add_argument(
        "--net-corrections", default=None, help="NET correction file",
    )
    parser.add_argument(
        "--target-dict", default=None, help="Target dictionary file",
    )
    parser.add_argument(
        "--mode", default="LinFit4", help="Fit mode",
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

    # Optionally, load NET corrections
    if args.net_corrections:
        net_corrections = pickle.load(open(args.net_corrections, "rb"))
    else:
        net_corrections = None

    # Optionally, load alternate source names
    if args.target_dict:
        target_dict = {}
        with open(args.target_dict) as fin:
            for line in fin:
                if line.strip().startswith("#"):
                    continue
                parts = line.split(",")
                target_dict[parts[0].strip()] = parts[1].strip()
    else:
        target_dict = None

    # Initialize the fitter

    bgmap = hp.read_map("npipe6v20_857_map.fits")
    # bgmap[bgmap < 0] = 0
    # bgmap[bgmap > 6] = 6

    fitter = tod2flux.FluxFitter(
        database,
        scan_length=args.scan_length_days,
        coord=args.coord,
        net_corrections=net_corrections,
        mode=args.mode,
        target_dict=target_dict,
        bgmap=bgmap,
    )

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
        fitter.detsets = False
        color_corrections, color_correction_errors = fitter.fit(
            target, pol=True, fname="results_{}.csv".format(target),
        )
        """
        fitter.detsets = True
        color_corrections2, color_correction_errors2 = fitter.fit(
            target,
            color_corrections=color_corrections,
            pol=True,
            fname="ccorrected_results_{}.csv".format(target),
        )
        """
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
