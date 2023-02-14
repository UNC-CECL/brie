#! /usr/bin/env python
from setuptools import find_packages, setup


def read(filename):
    with open(filename, "r", encoding="utf-8") as fp:
        return fp.read()


long_description = u"\n\n".join(
    [
        read("README.rst"),
        read("AUTHORS.rst"),
        read("CHANGES.rst"),
    ]
)


setup(
    name="brie",
    version="0.1.0.dev0",
    author="Katherine Anarde",
    author_email="kanarde@ncsu.edu",
    description="The Barrier Inlet Environment (BRIE) Model",
    long_description=long_description,
    python_requires=">=3.6",
    install_requires=open("requirements.txt", "r").read().splitlines(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=["earth science", "coast", "barrier inlet"],
    url="https://github.com/UNC-CECL/brie",
    license="MIT",
    packages=find_packages(),
)
