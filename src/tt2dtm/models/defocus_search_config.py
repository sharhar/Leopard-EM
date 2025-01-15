"""Serialization and validation of defocus search parameters for 2DTM."""

from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field


class DefocusSearchConfig(BaseModel):
    """Serialization and validation of defocus search parameters for 2DTM.

    Attributes
    ----------
    defocus_min : float
        Minimum searched defocus relative to average defocus ('defocus_u' and
        'defocus_v' in OpticsGroup) of micrograph in units of Angstroms.
    defocus_max : float
        Maximum searched defocus relative to average defocus ('defocus_u' and
        'defocus_v' in OpticsGroup) of micrograph in units of Angstroms.
    defocus_step : float
        Step size for defocus search in units of Angstroms.
    """

    defocus_min: Annotated[float, Field(..., gt=0.0)]
    defocus_max: Annotated[float, Field(..., gt=0.0)]
    defocus_step: Annotated[float, Field(..., gt=0.0)]

    @property
    def defocus_values(self) -> list[float]:
        """Gets a list of defocus values to search over."""
        vals = np.arange(
            self.defocus_min,
            self.defocus_max + self.defocus_step,
            self.defocus_step,
        )
        vals = vals.tolist()

        return vals  # type: ignore
