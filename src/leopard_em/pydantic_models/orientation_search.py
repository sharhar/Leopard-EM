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
    def euler_angles_offsets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Euler angle offsets to search over.

        For rotation along a specific axis, this primarily constrains to rotations
        around that axis using phi parameter, but also allows small variations in
        theta and psi if specified by the user.

        The rotation_axis_euler_angles field should contain [phi, theta, 0],
        where phi and theta define the direction of the rotation axis.
        The third value (psi) is ignored as it doesn't affect the axis direction.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            1. A tensor of shape (N, 3) representing the rotated Euler angles
            2. The original Euler angle grid used to generate the rotated angles
        """
        if not self.enabled:
            return torch.zeros((1, 3)), torch.zeros((1, 3))

        # If rotation axis is [0,0,0], perform a regular grid search
        if all(angle == 0.0 for angle in self.rotation_axis_euler_angles[:2]):
            phi_values = torch.arange(
                self.phi_min,
                self.phi_max + EPS,
                self.out_of_plane_step,
            )

            theta_values = torch.arange(
                self.theta_min,
                self.theta_max + EPS,
                self.out_of_plane_step,
            )

            psi_values = torch.arange(
                self.psi_min,
                self.psi_max + EPS,
                self.in_plane_step,
            )

            grid = torch.meshgrid(phi_values, theta_values, psi_values, indexing="ij")
            euler_angles_offsets = torch.stack(grid, dim=-1).reshape(-1, 3)
            return euler_angles_offsets, euler_angles_offsets

        # For a constrained search around a specific rotation axis
        else:
            # Create angle ranges for primary rotation and small adjustments
            phi_values = torch.arange(
                self.phi_min,
                self.phi_max + EPS,
                self.out_of_plane_step,
            )

            # Only create theta values if a range is specified (non-zero range)
            if self.theta_min != self.theta_max:
                theta_values = torch.arange(
                    self.theta_min,
                    self.theta_max + EPS,
                    self.out_of_plane_step,
                )
            else:
                theta_values = torch.tensor([0.0])

            # Only create psi values if a range is specified (non-zero range)
            if self.psi_min != self.psi_max:
                psi_values = torch.arange(
                    self.psi_min,
                    self.psi_max + EPS,
                    self.in_plane_step,
                )
            else:
                psi_values = torch.tensor([0.0])

            # Extract the rotation axis directly from phi and theta
            # (ignoring psi as it doesn't affect the axis direction)
            # 2. Define the rotation axis in Cartesian coordinates

            axis_phi, axis_theta = self.rotation_axis_euler_angles[:2]

            # Convert from degrees to radians
            axis_phi_tensor = torch.tensor(axis_phi, dtype=torch.float32)
            axis_theta_tensor = torch.tensor(axis_theta, dtype=torch.float32)
            axis_phi_rad = axis_phi_tensor * torch.pi / 180.0
            axis_theta_rad = axis_theta_tensor * torch.pi / 180.0

            # Convert to Cartesian coordinates (unit vector)
            axis = torch.tensor(
                [
                    torch.sin(axis_theta_rad) * torch.cos(axis_phi_rad),
                    torch.sin(axis_theta_rad) * torch.sin(axis_phi_rad),
                    torch.cos(axis_theta_rad),
                ]
            )

            # Ensure the axis points in the positive z direction
            if axis[2] < 0:
                axis = -axis
                # When flipping the axis, we need to adjust phi by 180 degrees
                # This ensures we're rotating in the same direction
                axis_phi = (axis_phi + 180) % 360
                axis_phi_tensor = torch.tensor(axis_phi, dtype=torch.float32)
                axis_phi_rad = axis_phi_tensor * torch.pi / 180.0

            # Create the grid of angles to search
            grid = torch.meshgrid(phi_values, theta_values, psi_values, indexing="ij")
            original_angles = torch.stack(grid, dim=-1).reshape(-1, 3)

            # Now process each combination of angles
            rotated_angles = []
            for phi, theta, psi in original_angles:
                # Convert to proper tensor if not already
                phi_tensor = (
                    phi.clone().detach()
                    if isinstance(phi, torch.Tensor)
                    else torch.tensor(phi, dtype=torch.float32)
                )

                # First apply rotation around the specified axis by phi
                main_rotation = roma.rotvec_to_rotmat(
                    axis * (phi_tensor * torch.pi / 180.0)
                )

                # Then apply small adjustments if specified (convert to matrices)
                adjustment = roma.euler_to_rotmat(
                    "ZYZ",
                    torch.tensor([0.0, theta, psi], dtype=torch.float32),
                    degrees=True,
                )

                # Combine rotations
                combined_rotation = roma.rotmat_composition((main_rotation, adjustment))

                # Convert back to Euler angles
                euler_angles = roma.rotmat_to_euler(
                    "ZYZ", combined_rotation, degrees=True
                )
                rotated_angles.append(euler_angles)

            # Stack all results
            euler_angles_rotated = torch.stack(rotated_angles)
            return euler_angles_rotated, original_angles

    '''
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
            self.phi_min,
            self.phi_max + EPS,
            self.out_of_plane_step,
        )

        theta_values = torch.arange(
            self.theta_min,
            self.theta_max + EPS,
            self.out_of_plane_step,
        )

        psi_values = torch.arange(
            self.psi_min,
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
        euler_angles_offsets_rotated = roma.rotmat_to_euler(
            "ZYZ", rot_matrix_batch, degrees=True
        ).to(torch.float32)
        return euler_angles_offsets_rotated, euler_angles_offsets
'''
