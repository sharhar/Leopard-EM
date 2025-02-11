"""Computational configuration for 2DTM."""

from typing import Annotated

from pydantic import BaseModel, Field


class ComputationalConfig(BaseModel):
    """Serialization of computational resources allocated for 2DTM.

    Attributes
    ----------
    gpu_ids : int | list[int]
        List of GPU IDs to use, defaults to [0].
    num_cpus : int
        Total number of CPUs to use, defaults to 1.
    """

    gpu_ids: Annotated[int | list[int], Field(...)] = [0]
    num_cpus: Annotated[int, Field(ge=1)] = 1
