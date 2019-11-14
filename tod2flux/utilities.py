""" Utility functions used in tod2flux pipelines
"""

import glob
import os
import sys


def find_files(path, detector_name):
    """ Find all small datasets for given detector
    """
    filenames = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if (
                filename.startswith("small_dataset")
                and filename.endswith(".fits")
                and detector_name in filename
            ):
                filenames.append(os.path.join(root, filename))
    return filenames
