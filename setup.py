#!/usr/bin/env python

"""The setup script."""

import sys
from setuptools import setup, find_packages
# import versioneer


# with open("requirements.txt") as f:
#     INSTALL_REQUIRES = f.read().strip().split("\n")

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

PYTHON_REQUIRES = '>=3.6'

description = ("Utilities for working with ctsm data")
setup(
    name="ctsm_py",
    description=description,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    maintainer="Joe Hamman",
    maintainer_email="jhamman@ucar.edu",
    url="https://github.com/ncar/ctsm_py",
    py_modules=['ctsm_py'],
    packages=find_packages(),
    python_requires=PYTHON_REQUIRES,
    license="Apache",
    keywords="ctsm_py",
    version='0.0.1',
)
