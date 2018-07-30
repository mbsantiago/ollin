#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ollin",
    version="0.0.1",
    author="CONABIO, Santiago Martínez",
    author_email="santiago.mbal@gmail.com",
    description="Animal Motion Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbsantiago/ollin",
    packages=setuptools.find_packages(),
    install_requires=[
        "six",
        "numpy",
        "numba",
        "scipy",
        "pystan",
        "backports.functools_lru_cache"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research"
    ),
)
