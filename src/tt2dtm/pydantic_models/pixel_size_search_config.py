"""Serialization and validation of pixel size search parameters for 2DTM."""

from typing import Annotated

import numpy as np
from pydantic import Field

from tt2dtm.pydantic_models.types import BaseModel2DTM


class PixelSizeSearchConfig(BaseModel2DTM):
    """Container for pixel size search configuration parameters.

    Attributes
    ----------
    enabled : bool
        Whether to enable pixel size search. Default is True.
    pixel_size_min : float
        Minimum pixel size to search, in Angstroms.
    pixel_size_max : float
        Maximum pixel size to search, in Angstroms.
    pixel_size_step : float
        Step size for pixel size search, in Angstroms.

    Properties
    ----------
    pixel_sizes : list[float]
        List of pixel sizes to search over based on held parameters.
    """

    enabled: bool = True
    pixel_size_min: Annotated[float, Field(..., gt=0.0)]
    pixel_size_max: Annotated[float, Field(..., gt=0.0)]
    pixel_size_step: Annotated[float, Field(..., gt=0.0)]

    @property
    def pixel_sizes(self) -> list[float]:
        """Gets a list of pixel sizes to search over."""
        vals = np.arange(
            self.pixel_size_min,
            self.pixel_size_max + self.pixel_size_step,
            self.pixel_size_step,
        )
        vals = vals.tolist()

        return vals  # type: ignore
