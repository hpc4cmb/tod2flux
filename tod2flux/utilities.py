""" Utility functions used in tod2flux pipelines
"""

import glob
import os
import sys

import astropy.io.fits as pf


def find_files(path, detector_name=None, pattern=None):
    """ Find all small datasets for given detector
    """
    filenames = {}
    for root, dirs, files in os.walk(path):
        for filename in files:
            if pattern is not None and pattern not in filename:
                continue
            if filename.startswith("small_dataset") and filename.endswith(".fits"):
                if detector_name is not None and detector_name not in filename:
                    continue
                full_path = os.path.join(root, filename)
                hdulist = pf.open(full_path, "readonly")
                detector = hdulist[1].header["detector"]
                if detector_name is not None and detector_name != detector:
                    continue
                if detector not in filenames:
                    filenames[detector] = []
                filenames[detector].append(full_path)
    return filenames
