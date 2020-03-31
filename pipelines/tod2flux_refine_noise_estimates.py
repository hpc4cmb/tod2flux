#!/usr/bin/env python

""" This pipeline loads a database of PSF fits and looks for a systematic
excess or deficit in chi-squared -- a sign or error in estimated noise level.
"""

import argparse
import os
import pickle
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
        "--mode", default="LinFit4", help="Fit mode",
    )
    parser.add_argument(
        "--out", default="net_corrections.pck", help="Correction file",
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

    # Process the database

    all_rchisq = {}

    for target, all_fits in database.targets.items():
        all_fits = database.targets[target]
        for fits in all_fits:
            for fit in fits:
                if args.mode not in fit.entries:
                    continue
                det = fit.detector
                rchisq = fit.entries[args.mode].rchisq
                if det not in all_rchisq:
                    all_rchisq[det] = []
                all_rchisq[det].append(rchisq)

    if len(all_rchisq) == 0:
        raise RuntimeError(
            "No entry in '{}' match mode = '{}'".format(args.database, args.mode)
        )

    net_corrections = {}

    for det in sorted(all_rchisq.keys()):
        rchisq = np.array(all_rchisq[det])
        print(
            "{} rchisq mean = {}, median = {}, sdev = {}, N = {}".format(
                det, np.mean(rchisq), np.median(rchisq), np.std(rchisq), rchisq.size
            )
        )
        net_corrections[det] = np.mean(rchisq)

    with open(args.out, "wb") as fout:
        pickle.dump(net_corrections, fout)

    print("Wrote corrections to '{}'".format(args.out))

    print("Pipeline completed in {:.2f} s".format(time.time() - t0), flush=True)

    return


if __name__ == "__main__":
    main()
