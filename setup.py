import os
from setuptools import setup, find_packages

packages = find_packages()
version = "0.0.0"

setup(
    name="fred",
    packages=packages,
    include_package_data=True,
    description="Python scripts useful for analyzing Rubin Observatory data",
    author="Fred Moolekamp",
    author_email="fred.moolekamp@gmail.com",
    url="https://github.com/fred3m/fred",
    version=version,
)
