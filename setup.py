# Copyright (c) Facebook, Inc. and its affiliates.
######################################################################
# \file setup.py
# \author Franziska Meier
#######################################################################
from setuptools import setup, find_packages

install_requires = ["higher", "pybullet", "matplotlib", "termcolor", "differentiable_robot_model", "jupyter"]

setup(
    name="l2l",
    author="Facebook AI Research",
    author_email="",
    version=1.0,
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)


