# Copyright (c) Facebook, Inc. and its affiliates.
######################################################################
# \file setup.py
# \author Franziska Meier
#######################################################################
from setuptools import setup, find_packages

install_requires = ["numpy", "higher", "matplotlib", "termcolor", "pybullet", "differentiable_robot_model", "dill", "jupyter"]

setup(
    name="ml3",
    author="Facebook AI Research",
    author_email="",
    version=1.0,
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)


