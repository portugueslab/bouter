#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("requirements_dev.txt") as f:
    requirements_dev = f.read().splitlines()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup_requirements = ["pytest-runner", "setuptools_scm"]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Vilim Stih, Hagar Lavian, Luigi Petrucco, Ot Prat @portugueslab",
    author_email="luigi.petrucco@gmail.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A package for behavioral analysis in python with Stytra data",
    install_requires=requirements,
    extras_require=dict(dev=requirements_dev),
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="bouter",
    name="bouter",
    packages=find_packages(include=["bouter", "bouter.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/portugueslab/bouter",
    version="0.2.0",
    zip_safe=False,
)
