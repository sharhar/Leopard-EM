"""Calculate the rotation axis for a pair of PDB structures."""

import math
import sys

import mmdf
import roma
import torch

sys.argv = [
    "rotate_pdbs.py",
    "3j77_aligned_SSU.pdb",
    "3j78_aligned_SSU.pdb",
    "rotation_axis.txt",
]


def calculate_rotation_matrix(
    coords1: torch.Tensor, coords2: torch.Tensor
) -> torch.Tensor:
    """Calculate rotation matrix from coords1 to coords2.

    Attributes
    ----------
    coords1: torch.Tensor
        The coordinates of the first PDB structure.
    coords2: torch.Tensor
        The coordinates of the second PDB structure.

    Returns
    -------
    torch.Tensor
        The rotation matrix.
    """
    # Calculate the covariance matrix
    H = coords1.T @ coords2

    # Perform SVD
    U, S, Vt = torch.linalg.svd(H)

    # Calculate rotation matrix
    R = Vt.T @ U.T

    # Handle reflection case
    det = torch.det(R)
    if det < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def extract_rotation_axis_angle(R: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Extract rotation axis and angle from rotation matrix.

    Attributes
    ----------
    R: torch.Tensor
        The rotation matrix.

    Returns
    -------
    tuple[torch.Tensor, float]
        The rotation axis and angle.
    """
    # Calculate angle from trace
    trace = torch.trace(R)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angle = torch.acos(cos_theta)

    # Handle special cases (very small angles or near 180 degrees)
    if torch.abs(angle) < 1e-6:
        return torch.tensor([0.0, 0.0, 1.0]), angle
    elif torch.abs(angle - math.pi) < 1e-6:
        diag = torch.diag(R) + 1
        axis_idx = torch.argmax(diag)
        axis = R[:, axis_idx].clone()
        axis = axis / torch.norm(axis)
        return axis, angle

    # Normal case - extract axis
    axis = torch.tensor([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    # Normalize the axis
    axis = axis / torch.norm(axis)

    return axis, angle


def calculate_axis_euler_angles(axis: torch.Tensor) -> tuple[float, float, float]:
    """Calculate Euler angles (ZYZ) that for the rotation axis.

    Attributes
    ----------
    axis: torch.Tensor
        The rotation axis.

    Returns
    -------
    tuple[float, float, float]
        The Euler angles.
    """
    # Z-axis unit vector
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    # Handle special cases
    if torch.norm(axis - z_axis) < 1e-6:
        return 0.0, 0.0, 0.0

    if torch.norm(axis + z_axis) < 1e-6:
        return 0.0, 180.0, 0.0

    # Calculate rotation matrix that aligns Z-axis with the given axis
    # First calculate the rotation axis (cross product)
    align_axis = torch.linalg.cross(z_axis, axis)

    # If align_axis is near zero, we need a different approach
    if torch.norm(align_axis) < 1e-6:
        return 0.0, 0.0, 0.0  # Z-axis is already aligned with rotation axis

    align_axis = align_axis / torch.norm(align_axis)

    # Calculate the angle between the axis and Z-axis (dot product)
    cos_angle = torch.dot(axis, z_axis)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

    # Build rotation matrix using Roma
    # Create a rotation around the align_axis by the calculated angle
    rotmat = roma.rotvec_to_rotmat(align_axis * angle)

    # Convert rotation matrix to Euler angles using RoMA
    euler_angles = roma.rotmat_to_euler(
        convention="ZYZ", rotmat=rotmat, as_tuple=False, degrees=True
    )

    # Extract and normalize angles to requested ranges
    phi = euler_angles[0].item() % 360
    if phi < 0:
        phi += 360

    theta = euler_angles[1].item() % 360
    if theta > 180:
        theta = 360 - theta
        phi = (phi + 180) % 360
        psi = (euler_angles[2].item() + 180) % 360
    else:
        psi = euler_angles[2].item() % 360

    if psi < 0:
        psi += 360

    return phi, theta, psi


def main() -> None:
    """Calculate rotation axis for a pair of PDB structures."""
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <pdb_file1> <pdb_file2> <output_file>")
        sys.exit(1)

    pdb_file1 = sys.argv[1]
    pdb_file2 = sys.argv[2]
    output_file = sys.argv[3]

    # Parse PDB files
    df1 = mmdf.read(pdb_file1)
    df2 = mmdf.read(pdb_file2)

    # Extract coordinates
    coords1 = torch.tensor(df1[["x", "y", "z"]].values, dtype=torch.float32)
    coords2 = torch.tensor(df2[["x", "y", "z"]].values, dtype=torch.float32)

    # Center coordinates
    centroid1 = coords1.mean(dim=0)
    centroid2 = coords2.mean(dim=0)
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2

    # Calculate rotation matrix
    rotation_matrix = calculate_rotation_matrix(coords1_centered, coords2_centered)

    # Extract rotation axis and angle
    rotation_axis, rotation_angle = extract_rotation_axis_angle(rotation_matrix)

    # Calculate Euler angles for the rotation axis
    phi, theta, psi = calculate_axis_euler_angles(rotation_axis)

    # Output results
    angle_degrees = float(rotation_angle) * 180 / math.pi

    print("\nRotation Analysis Results:")
    print(
        f"Rotation Axis: [{rotation_axis[0]:.6f}, "
        f"{rotation_axis[1]:.6f}, {rotation_axis[2]:.6f}]"
    )
    print(f"Rotation Angle: {angle_degrees:.2f} degrees")

    print("\nEuler Angles of the rotation axis (ZYZ convention):")
    print(f"  Phi:   {phi:.2f} degrees (range 0-360)")
    print(f"  Theta: {theta:.2f} degrees (range 0-180)")
    print(f"  Psi:   {psi:.2f} degrees (range 0-360)")

    # Write results to output file
    with open(output_file, "w") as f:
        f.write("# PDB Rotation Analysis Results\n\n")
        f.write(f"Source PDB: {pdb_file1}\n")
        f.write(f"Target PDB: {pdb_file2}\n")
        f.write(f"Number of atoms: {len(df1)}\n\n")

        f.write("## Rotation Parameters\n")
        f.write(
            f"Axis: {rotation_axis[0]:.6f} "
            f"{rotation_axis[1]:.6f} {rotation_axis[2]:.6f}\n"
        )
        f.write(f"Angle: {angle_degrees:.6f} degrees\n\n")

        f.write("## Euler Angles of the rotation axis (ZYZ convention)\n")
        f.write(f"Phi: {phi:.6f} degrees\n")
        f.write(f"Theta: {theta:.6f} degrees\n")
        f.write(f"Psi: {psi:.6f} degrees\n")

    print(f"Rotation analysis written to: {output_file}")


if __name__ == "__main__":
    main()
