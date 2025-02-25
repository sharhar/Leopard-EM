"""Two-Dimensional Template Matching (2DTM) written in Python."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("leopard_em")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = ["Josh Dickerson", "Matthew Giammar"]
__email__ = ["jdickerson@berkeley.edu", "matthew_giammar@berkeley.edu"]
