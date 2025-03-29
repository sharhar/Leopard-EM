"""Serialization and validation of orientation search parameters for 2DTM."""

import re
from typing import Annotated, Literal

import roma
import torch
from pydantic import Field
from torch_so3 import angular_ranges, get_uniform_euler_angles

from leopard_em.pydantic_models.types import BaseModel2DTM

EPS = 1e-6


class OrientationSearchConfig(BaseModel2DTM):
    """Serialization and validation of orientation search parameters for 2DTM.

    The angles -- phi, theta, and psi -- represent Euler angles in the 'ZYZ'
    convention.

    This model effectively acts as a connector into the
    `torch_so3.uniform_so3_sampling.get_uniform_euler_angles` function from the
    [torch-so3](https://github.com/teamtomo/torch-so3) package.

    TODO: Add parameters for template symmetry.

    TODO: Implement indexing to get the i-th or range of orientations in the
    search space (need to be ordered).

    Attributes
    ----------
    orientation_sampling_method : str
        Method for sampling orientations. Default is 'Hopf Fibration'.
        Currently only 'Hopf Fibration' is supported.
    template_symmetry : str
        Symmetry group of the template. Default is 'C1'.
        Currently only 'C1' is supported.
    phi_min : float
        Minimum value for the phi angle in degrees.
    phi_max : float
        Maximum value for the phi angle in degrees.
    theta_min : float
        Minimum value for the theta angle in degrees.
    theta_max : float
        Maximum value for the theta angle in degrees.
    psi_min : float
        Minimum value for the psi angle in degrees.
    psi_max : float
        Maximum value for the psi angle in degrees.
    in_plane_angular_step : float
        Angular step size for in-plane rotations in degrees. Must be greater
        than 0.
    out_of_plane_angular_step : float
        Angular step size for out-of-plane rotations in degrees. Must be
        greater than 0.
    """

    # TODO: Particle symmetry options

    in_plane_step: Annotated[float, Field(ge=0.0)] = 1.5
    out_of_plane_step: Annotated[float, Field(ge=0.0)] = 2.5
    phi_min: float = 0.0
    phi_max: float = 360.0
    theta_min: float = 0.0
    theta_max: float = 180.0
    psi_min: float = 0.0
    psi_max: float = 360.0
    base_grid_method: Literal["uniform", "healpix"] = "uniform"
    symmetry: str = "C1"

    @property
    def euler_angles(self) -> torch.Tensor:
        """Returns the Euler angles ('ZYZ' convention) to search over.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 3) where N is the number of orientations to
            search over. The columns represent the psi, theta, and phi angles
            respectively.
        """
        if self.symmetry != "C1":
            # If the user has not specified the ranges, use the symmetry group
            if (
                self.phi_min == 0.0
                and self.phi_max == 360.0
                and self.theta_min == 0.0
                and self.theta_max == 180.0
                and self.psi_min == 0.0
                and self.psi_max == 360.0
            ):
                # Extract symmetry group and order
                match = re.match(r"([A-Za-z]+)(\d*)", self.symmetry)
                if not match:
                    raise ValueError(f"Invalid symmetry format: {self.symmetry}")
                sym_group = match.group(1)
                sym_order = int(match.group(2)) if match.group(2) else 1
                (
                    self.phi_min,
                    self.phi_max,
                    self.theta_min,
                    self.theta_max,
                    self.psi_min,
                    self.psi_max,
                ) = angular_ranges(sym_group, sym_order)

        return get_uniform_euler_angles(
            in_plane_step=self.in_plane_step,
            out_of_plane_step=self.out_of_plane_step,
            phi_min=self.phi_min,
            phi_max=self.phi_max,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            psi_min=self.psi_min,
            psi_max=self.psi_max,
            base_grid_method=self.base_grid_method,
        )


class RefineOrientationConfig(BaseModel2DTM):
    """Serialization and validation of orientation refinement parameters.

    Angles will be sampled from [-coarse_step, coarse_step] in increments of
    'fine_step' for the orientation refinement search.

    Attributes
    ----------
    orientation_sampling_method : str
        Method for sampling orientations. Default is 'Hopf Fibration'.
        Currently only 'Hopf Fibration' is supported.
    template_symmetry : str
        Symmetry group of the template. Default is 'C1'.
        Currently only 'C1' is supported.
    in_plane_angular_step_coarse : float
        Angular step size for in-plane rotations in degrees for previous, coarse search.
        This corresponds to the 'OrientationSearchConfig.in_plane_angular_step' value
        for the match template program. Must be greater than or equal to 0.
    in_plane_angular_step_fine : float
        Angular step size for in-plane rotations in degrees for current, fine search.
        Must be greater than or equal to 0.
    out_of_plane_angular_step_coarse : float
        Angular step size for out-of-plane rotations in degrees for previous, coarse
        search. This corresponds to the
        'OrientationSearchConfig.out_of_plane_angular_step' value for the match template
        program. Must be greater than or equal to 0.
    out_of_plane_angular_step_fine : float
        Angular step size for out-of-plane rotations in degrees for current, fine
        search. Must be greater than or equal to 0.

    """

    enabled: bool = True
    in_plane_angular_step_coarse: Annotated[float, Field(..., ge=0.0)] = 1.5
    in_plane_angular_step_fine: Annotated[float, Field(..., ge=0.0)] = 0.1
    out_of_plane_angular_step_coarse: Annotated[float, Field(..., ge=0.0)] = 2.5
    out_of_plane_angular_step_fine: Annotated[float, Field(..., ge=0.0)] = 0.25

    @property
    def euler_angles_offsets(self) -> torch.Tensor:
        """Return the Euler angle offsets to search over.

        Note that this method uses a uniform grid search which approximates SO(3) space
        well when the angular ranges are small (e.g. ±2.5 degrees).

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 3) where N is the number of orientations to
            search over. The columns represent the phi, theta, and psi angles,
            respectively, in the 'ZYZ' convention.
        """
        if not self.enabled:
            return torch.zeros((1, 3))

        phi_values = torch.arange(
            -self.out_of_plane_angular_step_coarse,
            self.out_of_plane_angular_step_coarse + EPS,
            self.out_of_plane_angular_step_fine,
        )

        theta_values = torch.arange(
            0.0,
            self.out_of_plane_angular_step_coarse + EPS,
            self.out_of_plane_angular_step_fine,
        )

        psi_values = torch.arange(
            -self.in_plane_angular_step_coarse,
            self.in_plane_angular_step_coarse + EPS,
            self.in_plane_angular_step_fine,
        )

        grid = torch.meshgrid(phi_values, theta_values, psi_values, indexing="ij")
        euler_angles_offsets = torch.stack(grid, dim=-1).reshape(-1, 3)

        return euler_angles_offsets


class ConstrainedOrientationConfig(BaseModel2DTM):
    """Serialization and validation of constrained orientation parameters.

    Attributes
    ----------
    enabled: bool
        Whether to enable constrained orientation search.
    in_plane_step: float
        Angular step size for in-plane rotations in degrees.
        Must be greater than or equal to 0.
    out_of_plane_step: float
        Angular step size for out-of-plane rotations in degrees.
        Must be greater than or equal to 0.
    rotation_axis_euler_angles: list[float]
        List of Euler angles (phi, theta, psi) for the rotation axis.
    phi_min: float
        Minimum value for the phi angle in degrees.
    phi_max: float
        Maximum value for the phi angle in degrees.
    theta_min: float
        Minimum value for the theta angle in degrees.
    theta_max: float
        Maximum value for the theta angle in degrees.
    psi_min: float
        Minimum value for the psi angle in degrees.
    psi_max: float
        Maximum value for the psi angle in degrees.
    """

    enabled: bool = True
    in_plane_step: float = 1.5
    out_of_plane_step: float = 2.5
    rotation_axis_euler_angles: list[float] = Field(default=[0.0, 0.0, 0.0])
    phi_min: float = 0.0
    phi_max: float = 360.0
    theta_min: float = 0.0
    theta_max: float = 180.0
    psi_min: float = 0.0
    psi_max: float = 360.0

    @property
    def euler_angles_offsets(self) -> torch.Tensor:
        """Return the Euler angle offsets to search over.

        Note that this method uses a uniform grid search which approximates SO(3) space
        well when the angular ranges are small.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 3) where N is the number of orientations to
            search over. The columns represent the phi, theta, and psi angles,
            respectively, in the 'ZYZ' convention.
        """
        if not self.enabled:
            return torch.zeros((1, 3))

        phi_values = torch.arange(
            -self.phi_min,
            self.phi_max + EPS,
            self.out_of_plane_step,
        )

        theta_values = torch.arange(
            self.theta_min,
            self.theta_max + EPS,
            self.out_of_plane_step,
        )

        psi_values = torch.arange(
            -self.psi_min,
            self.psi_max + EPS,
            self.in_plane_step,
        )

        grid = torch.meshgrid(phi_values, theta_values, psi_values, indexing="ij")
        euler_angles_offsets = torch.stack(grid, dim=-1).reshape(-1, 3)
        # Convert to rotation matrix
        rot_z_matrix = roma.euler_to_rotmat(
            "ZYZ",
            euler_angles_offsets,
            degrees=True,
            device=euler_angles_offsets.device,
        ).to(torch.float32)
        # Apply rotation to the template
        rot_axis_matrix = roma.euler_to_rotmat(
            "ZYZ",
            torch.tensor(self.rotation_axis_euler_angles),
            degrees=True,
            device=euler_angles_offsets.device,
        ).to(torch.float32)

        rot_matrix_batch = roma.rotmat_composition(
            sequence=(rot_axis_matrix, rot_z_matrix, rot_axis_matrix.transpose(-1, -2))
        )

        # Convert back to Euler angles
        euler_angles_offsets = roma.rotmat_to_euler(
            "ZYZ", rot_matrix_batch, degrees=True, device=euler_angles_offsets.device
        ).to(torch.float32)

        return euler_angles_offsets
