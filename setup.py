#!/usr/bin/env python

import glob
import os
import re
import unittest

from Cython.Build import cythonize
from setuptools import setup, Extension
from setuptools.command.test import test as TestCommand

import numpy as np

# extensions to builde

ext_kernels = Extension(
    'tod2flux.kernels',
    include_dirs=[np.get_include()],
    sources=['tod2flux/kernels.pyx'],
)

extensions = cythonize([ext_kernels])

# scripts to install

scripts = glob.glob('pipelines/*.py')

# run unit tests


class PTestCommand(TestCommand):

    def __init__(self, *args, **kwargs):
        super(PTestCommand, self).__init__(*args, **kwargs)

    def initialize_options(self):
        TestCommand.initialize_options(self)

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_suite = True

    def run(self):
        loader = unittest.TestLoader()
        runner = unittest.TextTestRunner(verbosity=2)
        suite = loader.discover('tests', pattern='test_*.py', top_level_dir='.')
        runner.run(suite)

# set it all up


setup(
    name='tod2flux',
    provides=['tod2flux'],
    version="0.1",
    description="Tools for estimating point source flux densities from TOD",
    author="Reijo Keskitalo and Graca Rocha",
    author_email='rtkeskitalo@lbl.gov',
    url='https://github.com/hpc4cmb/tod2flux',
    packages=['tod2flux', 'tod2flux.planck'],
    ext_modules=extensions,
    scripts=scripts,
    license='BSD',
    requires=['Python (>3.4.0)', ],
    cmdclass={'test': PTestCommand}
)
