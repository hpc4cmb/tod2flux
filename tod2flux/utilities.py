""" Utility functions used in tod2flux pipelines
"""

from datetime import datetime, timezone, timedelta
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


def to_UTC(t):
    # Convert UNIX time stamp to a date string
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def to_date(t):
    # Convert UNIX time stamp to a date string
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")


def to_JD(t):
    # Unix time stamp to Julian date
    # (days since -4712-01-01 12:00:00 UTC)
    return t / 86400.0 + 2440587.5


def to_MJD(t):
    # Convert Unix time stamp to modified Julian date
    # (days since 1858-11-17 00:00:00 UTC)
    return to_JD(t) - 2400000.5


def to_DJD(t):
    # Convert Unix time stamp to Dublin Julian date
    # (days since 1899-12-31 12:00:00)
    # This is the time format used by PyEphem
    return to_JD(t) - 2415020


def DJDtoUNIX(djd):
    # Convert Dublin Julian date to a UNIX time stamp
    return ((djd + 2415020) - 2440587.5) * 86400.0
