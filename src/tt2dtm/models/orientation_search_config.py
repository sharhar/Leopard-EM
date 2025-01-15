"""Serialization and validation of orientation search parameters for 2DTM."""

from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class OrientationSearchConfig(BaseModel):
    """Serialization and validation of orientation search parameters for 2DTM.

    Attributes
    ----------
    orientation_sampling_method : str
        Method for sampling orientations. Default is 'Hopf Fibration'.
        Currently only 'Hopf Fibration' is supported.
    template_symmetry : str
        Symmetry group of the template. Default is 'C1'.
    psi_range : tuple[float, float]
        Range of psi angles in degrees. Default is [0, 360).
    theta_range : tuple[float, float]
        Range of theta angles in degrees. Default is [0, 180).
    phi_range : tuple[float, float]
        Range of phi angles in degrees. Default is [0, 360).
    in_plane_angular_step : float
        Angular step size for in-plane rotations in degrees. Must be greater
        than 0.
    out_of_plane_angular_step : float
        Angular step size for out-of-plane rotations in degrees. Must be
        greater than 0.
    """

    orientation_sampling_method: Annotated[str, Field(default="Hopf Fibration")] = (
        "Hopf Fibration"
    )
    template_symmetry: Annotated[str, Field(default="C1")] = "C1"
    psi_range: Annotated[tuple[float, float], Field(...)] = (0.0, 360.0)
    theta_range: Annotated[tuple[float, float], Field(...)] = (0.0, 180.0)
    phi_range: Annotated[tuple[float, float], Field(...)] = (0.0, 360.0)
    in_plane_angular_step: Annotated[float, Field(..., gt=0.0)]
    out_of_plane_angular_step: Annotated[float, Field(..., gt=0.0)]

    @field_validator("orientation_sampling_method")
    def validate_orientation_sampling_method(cls, value):  # type: ignore
        """Validate from allowed orientation sampling methods."""
        if value != "Hopf Fibration":
            raise ValueError("Currently only supports 'Hopf Fibration'.")

        return value
