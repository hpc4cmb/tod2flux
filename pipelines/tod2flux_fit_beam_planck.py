#!/usr/bin/env python

""" This pipeline fits small dataset files with instrumental beam to
measure the apparent flux density through one detector.
"""

import argparse
import os
import sys
import time

import numpy as np

import tod2flux
import tod2flux.planck


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fit Planck beams to data", fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "--input-path", required=True, help="Path to search small dataset files from",
    )
    parser.add_argument(
        "--detector", required=True, help="Name of the detector to fit",
    )
    parser.add_argument(
        "--database", default="fluxes.db", help="Name of the fit database",
    )
    args = parser.parse_args()
    print("All parameters:")
    print(args, flush=True)
    return args


def main():
    t0 = time.time()
    args = parse_arguments()

    # Load the detector

    detector = tod2flux.planck.Detector(args.detector)
    print(detector)

    # Instantiate the fitter

    fitter = tod2flux.DetectorFitter(detector)

    # Load the fit database

    database = tod2flux.Database(args.database)

    # List the small dataset files

    filenames = tod2flux.find_files(args.input_path, detector.name)
    print(
        "Found {} files for {} under {}".format(
            len(filenames), detector.name, args.input_path
        ),
        flush=True,
    )

    # Loop over the files

    for filename in filenames:

        # Load the dataset

        dataset = tod2flux.Dataset(filename, psi_pol=detector.psi_pol)
        print(dataset)

        # Fit the detector over the data

        fit = fitter.fit(dataset)

        # Write the fit into the database

        database.enter(fit)

        break # DEBUG

    # Save the database

    database.save()

    print("Pipeline completed in {:.2f} s".format(time.time() - t0), flush=True)

    return


if __name__ == "__main__":
    main()
