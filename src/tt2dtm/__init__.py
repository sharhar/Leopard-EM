"""2DTM written in pyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tt2DTM")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = ["Josh Dickerson", "Matthew Giammar"]
__email__ = ["jdickerson@berkeley.edu", "matthew_giammar@berkeley.edu"]
