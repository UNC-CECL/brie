#! /usr/bin/env python
from setuptools import find_packages, setup

import versioneer


setup(
    name="brie",
    version=versioneer.get_version(),
    author="Katherine Anarde",
    author_email="anardek@gmail.com",
    description="The Barrier Inlet Environment model",
    long_description=open("README.rst", encoding="utf-8").read(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    url="https://github.com/mcflugen/brie",
    license="MIT",
    install_requires=open("requirements.txt", "r").read().splitlines(),
    packages=find_packages(),
    cmdclass=versioneer.get_cmdclass(),
)
