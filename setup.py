#!/usr/bin/env python
######################################################################
# \file setup.py
#######################################################################
from setuptools import setup, find_packages

__author__ = "Giovanni Sutanto"

install_requires = [
    "spyder", "rtree", "numpy", "matplotlib", "scipy"
]
dependency_links = [
]

setup(
    name="dmp",
    version="1.0",
    author="Giovanni Sutanto",
    author_email="gsutanto@alumni.usc.edu",
    description="Dynamic Movement Primitives",
    packages=find_packages(),
    install_requires=install_requires,
    dependency_links=dependency_links,
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.7',
)
