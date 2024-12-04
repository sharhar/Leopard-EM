"""2DTM in pyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tt2DTM")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"
