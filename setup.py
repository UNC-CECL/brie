#! /usr/bin/env python
from setuptools import find_packages, setup


setup(
    name="brie",
    version="0.1.0.dev0",
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
    url="https://github.com/UNC-CECL/brie",
    license="MIT",
    install_requires=open("requirements.txt", "r").read().splitlines(),
    packages=find_packages(),
)
