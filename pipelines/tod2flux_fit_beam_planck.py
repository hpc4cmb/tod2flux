#!/usr/bin/env python

""" This pipeline fits small dataset files with instrumental beam to
measure the apparent flux density through one detector.
"""

import argparse
from datetime import datetime
import os
import sys
import time

import healpy as hp
import numpy as np

import tod2flux
import tod2flux.planck

from mpi4py import MPI


def parse_arguments(comm):
    parser = argparse.ArgumentParser(
        description="Fit Planck beams to data", fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "--input-path", required=True, help="Path to search small dataset files from",
    )
    parser.add_argument(
        "--input-pattern", required=False, help="Filter dataset files",
    )
    parser.add_argument(
        "--detector", required=False, help="Name of the detector to fit",
    )
    parser.add_argument(
        "--fit-radius-fwhm", default=2, type=np.int, help="Fit radius in multiples of FWHM",
    )
    parser.add_argument(
        "--database", default="fluxes.pck", help="Name of the fit database",
    )
    parser.add_argument(
        "--background", required=False, help="Background map to subtract",
    )
    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help="Treat all datasets as new",
        dest="overwrite",
    )
    parser.add_argument(
        "--no-overwrite",
        required=False,
        action="store_false",
        help="Skip datasets that already exist in the database",
        dest="overwrite",
    )
    parser.set_defaults(overwrite=False)

    args = parser.parse_args()
    if comm.rank == 0:
        print("All parameters:")
        print(args, flush=True)

    return args


def main():
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print("Running with {} MPI tasks".format(comm.size), flush=True)

    t0 = time.time()
    args = parse_arguments(comm)

    # Load the fit database

    if comm.rank == 0:
        database = tod2flux.Database(args.database)
    else:
        database = None
    database = comm.bcast(database)

    # List the small dataset files

    if comm.rank == 0:
        filenames = tod2flux.find_files(
            args.input_path, args.detector, args.input_pattern
        )
    else:
        filenames = None
    filenames = comm.bcast(filenames)

    # Optionally load the background map

    if args.background:
        bg = hp.read_map(args.background)
    else:
        bg = None

    # Loop over the files

    ifile = -1

    for detector_name in sorted(filenames.keys()):
        detector_files = filenames[detector_name]
        my_fits = []

        if comm.rank == 0:
            print(
                "Found {} files for {} under {}".format(
                    len(detector_files), detector_name, args.input_path
                ),
                flush=True,
            )

        # Instantiate detector

        detector = tod2flux.planck.Detector(detector_name)
        print("detector =", detector)

        # Determine fit radius

        fit_radius = args.fit_radius_fwhm * detector.fwhm_arcmin

        # Instantiate fitter

        fitter = tod2flux.DetectorFitter(detector, fit_radius_arcmin=fit_radius)
        print(fitter)

        # Fit each file

        for filename in detector_files:
            print("filename =", filename)

            if filename in database:
                if args.overwrite:
                    if comm.rank == 0:
                        print("Overwriting {} in database".format(filename), flush=True)
                else:
                    if comm.rank == 0:
                        print("{} is already processed".format(filename), flush=True)
                    continue

            ifile += 1
            if ifile % comm.size != comm.rank:
                continue

            # Load the dataset

            print("Loading", filename)
            # Planck time stamps count from 1958-01-01 00:00:00
            dataset = tod2flux.Dataset(
                filename, time_offset=datetime(1958, 1, 1).timestamp(), background=bg
            )
            print(dataset)

            # Fit the detector over the data

            fits = fitter.fit(dataset)

            # Write the fit into the database

            if len(fits) != 0:
                my_fits.append(fits)

        all_fits = comm.gather(my_fits)
        if comm.rank == 0:
            for fits in all_fits:
                for fit in fits:
                    database.enter(fit)
            # Save the database after every detector
            database.save()

    if comm.rank == 0:
        print("Pipeline completed in {:.2f} s".format(time.time() - t0), flush=True)

    return


if __name__ == "__main__":
    main()
