#!/bin/bash

# Use this script to convert all .ipynb files in the examples directory to .py files for rendering.
# There are some basic formatting rules for properly rendering notebooks in the gallery, for now
# the conversions are manual but could be automated in the future.
# The following rules must be followed for the conversion to work properly:

# 1) All notebook files must have a docstring at the first code cell.
# For example:
# >>> """This is a demo docstring"""
# >>> 
# >>> import numpy as np
# >>> ...

# 2) Notebooks which should be dynamically generated TODO

jupytext --set-formats ipynb,py:percent --to py basic_configuration.ipynb