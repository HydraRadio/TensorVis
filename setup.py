from setuptools import setup, find_packages
import glob
import os
import sys

VERSION = '0.0.9'

setup_args = {
    "name": "TensorVis",
    "author": "Phil Bull",
    "url": "https://github.com/philbull/TensorVis",
    "license": "MIT",
    "description": "Radio interferometer visibility simulator written for TensorFlow.",
    "package_dir": {"TensorVis": "TensorVis"},
    "packages": find_packages(),
    "include_package_data": True,
    "install_requires": [
        'numpy>=1.19',
        'scipy',
        'pyuvdata',
        'tensorflow>=2.2',
        "future"
    ],
    "version": VERSION,
    "zip_safe": False,
}


if __name__ == "__main__":
    setup(*(), **setup_args)
