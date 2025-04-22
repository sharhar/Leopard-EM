"""Calculates the center vector between two PDB structures."""

import argparse

import mmdf
import numpy as np
import pandas as pd
import roma
import torch


def calculate_mean_position(df: pd.DataFrame) -> torch.Tensor:
    """
    Calculate the mean position of a PDB structure.

    Attributes
    ----------
    df: pd.DataFrame
        The mmdf DataFrame with the PDB structure.

    Returns
    -------
    torch.Tensor
        The mean position as a 3D vector [x, y, z].
    """
    # Extract coordinates and calculate mean
    coords = torch.tensor(df[["x", "y", "z"]].values, dtype=torch.float32)
    mean_pos = coords.mean(dim=0)

    return mean_pos


def rotate_vector(vector: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector using a rotation matrix.

    Attributes
    ----------
    vector: torch.Tensor
        The vector to rotate.
    rotation_matrix: torch.Tensor
        The rotation matrix to use.

    Returns
    -------
        Rotated vector
    """
    # Convert to tensor if not already
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32)

        # Ensure matching types between rotation_matrix and vector
    if rotation_matrix.dtype != vector.dtype:
        rotation_matrix = rotation_matrix.to(dtype=vector.dtype)

    # Apply rotation
    rotated_vector = rotation_matrix @ vector

    return rotated_vector


def setup_argparse() -> argparse.Namespace:
    """
    Set up and parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Calculate vector between PDB structures"
    )
    parser.add_argument("pdb_file1", help="PDB file of larger structure")
    parser.add_argument("pdb_file2", help="PDB file of smaller structure")
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=5,
        help="Number of random rotations to test",
    )
    parser.add_argument(
        "--output-file", default="defocus_analysis.txt", help="Output analysis file"
    )
    return parser.parse_args()


def calculate_relative_vectors(pdb_file1: str, pdb_file2: str) -> dict:
    """
    Calculate the relative position and orientation vectors between two PDB structures.

    Parameters
    ----------
    pdb_file1 : str
        Path to the first PDB file
    pdb_file2 : str
        Path to the second PDB file

    Returns
    -------
    dict
        Dictionary containing relative vector data including:
        - df1, df2: DataFrames for both PDB files
        - vector: Vector from PDB1 to PDB2
        - euler_angles: Phi, Theta, Psi angles
        - z_diff: Z height difference
        - defocus_description: Human-readable defocus description
    """
    # Parse PDB files using mmdf
    df1 = mmdf.read(pdb_file1)
    df2 = mmdf.read(pdb_file2)

    print(f"File 1: {pdb_file1} - {len(df1)} atoms")
    print(f"File 2: {pdb_file2} - {len(df2)} atoms")

    # Calculate mean positions at default orientation (0,0,0)
    mean_pos1 = calculate_mean_position(df1)
    mean_pos2 = calculate_mean_position(df2)

    # Calculate vector from PDB1 to PDB2
    vector = mean_pos2 - mean_pos1

    # Convert vector to Euler angles
    phi, theta, psi = roma.rotvec_to_euler(
        convention="ZYZ", rotvec=vector, degrees=True, as_tuple=True
    )

    # Calculate Z-height difference (defocus)
    z_diff = vector[2].item()
    defocus_description = (
        f"{abs(z_diff):.2f} Angstroms {'below' if z_diff < 0 else 'above'}"
    )

    # Print initial results
    print("\nInitial Analysis:")
    print(
        f"Vector from PDB1 to PDB2: [{vector[0]:.6f}, {vector[1]:.6f}, {vector[2]:.6f}]"
    )
    print(
        f"Vector Euler angles (ZYZ): Phi={phi:.2f}°, Theta={theta:.2f}°, Psi={psi:.2f}°"
    )
    print(f"Z-height difference (defocus): {defocus_description}")

    return {
        "df1": df1,
        "df2": df2,
        "vector": vector,
        "euler_angles": (phi, theta, psi),
        "z_diff": z_diff,
        "defocus_description": defocus_description,
    }


def generate_rotation_samples(num_rotations: int) -> torch.Tensor:
    """
    Generate a set of random rotations for testing.

    Parameters
    ----------
    num_rotations : int
        Number of random rotations to generate

    Returns
    -------
    torch.Tensor
        Tensor containing rotation angles (Euler ZYZ convention)
    """
    # Generate random Euler angles for testing
    if num_rotations > 1:
        # Generate all random angles with a single call
        random_rotations = np.random.uniform(-180, 180, size=(num_rotations - 1, 3))
        # Adjust theta values to be between 0 and 180
        random_rotations[:, 1] = np.abs(random_rotations[:, 1]) % 180
        # Start with default orientation [0,0,0] and concatenate random rotations
        rotations = np.vstack(([0.0, 0.0, 0.0], random_rotations))
    else:
        # Just use the default orientation if num_rotations is 1
        rotations = np.array([[0.0, 0.0, 0.0]])

    # Convert to tensor for batch processing
    return torch.tensor(rotations, dtype=torch.float32)


def process_rotations(vector: torch.Tensor, rotations_tensor: torch.Tensor) -> list:
    """
    Process each rotation and calculate the resulting defocus.

    Parameters
    ----------
    vector : torch.Tensor
        The original vector between structures
    rotations_tensor : torch.Tensor
        Tensor containing rotation angles to test

    Returns
    -------
    list
        List of dictionaries with defocus results for each rotation
    """
    print("\nDefocus changes for different rotations:")
    defocus_results = []

    for i, euler in enumerate(rotations_tensor):
        # Convert degrees to radians using np.deg2rad
        phi_rad, theta_rad, psi_rad = (np.deg2rad(angle.item()) for angle in euler)

        # Create rotation matrix using RoMA (ZYZ intrinsic convention)
        euler_angles = torch.tensor([phi_rad, theta_rad, psi_rad])
        rotation_matrix = roma.rotvec_to_rotmat(
            roma.euler_to_rotvec(convention="ZYZ", angles=euler_angles)
        )

        # Rotate the vector
        rotated_vector = rotate_vector(vector, rotation_matrix)

        # Extract new z-component (defocus)
        new_z_diff = rotated_vector[2].item()
        new_defocus = (
            f"{abs(new_z_diff):.2f} Angstroms {'below' if new_z_diff < 0 else 'above'}"
        )

        print(
            f"Rotation #{i+1} - Euler({euler[0]:.2f}°, {euler[1]:.2f}°, "
            f"{euler[2]:.2f}°): Defocus = {new_defocus}"
        )

        defocus_results.append(
            {
                "rotation": i + 1,
                "euler_angles": [e.item() for e in euler],
                "defocus": new_z_diff,
                "description": new_defocus,
            }
        )

    return defocus_results


def write_results_to_file(
    output_file: str,
    pdb_file1: str,
    pdb_file2: str,
    vector_data: dict,
    defocus_results: list,
) -> None:
    """
    Write analysis results to output file.

    Parameters
    ----------
    output_file : str
        Path to output file
    pdb_file1 : str
        Path to first PDB file
    pdb_file2 : str
        Path to second PDB file
    vector_data : dict
        Dictionary with vector data from calculate_relative_vectors
    defocus_results : list
        List of defocus results from process_rotations
    """
    vector = vector_data["vector"]
    phi, theta, psi = vector_data["euler_angles"]
    defocus_description = vector_data["defocus_description"]

    with open(output_file, "w") as f:
        f.write("# PDB Vector and Defocus Analysis\n\n")
        f.write(f"Source PDB 1: {pdb_file1}\n")
        f.write(f"Source PDB 2: {pdb_file2}\n\n")

        f.write("## Initial Vector Analysis\n")
        f.write(
            f"Vector PDB1-PDB2: [{vector[0]:.6f}, {vector[1]:.6f}, {vector[2]:.6f}]\n"
        )
        f.write(
            f"Vector Eulers (ZYZ): Phi={phi:.2f}°, Theta={theta:.2f}°, Psi={psi:.2f}°\n"
        )
        f.write(f"Z-height difference (defocus): {defocus_description}\n\n")

        f.write("## Defocus changes for different rotations\n")
        for result in defocus_results:
            euler = result["euler_angles"]
            f.write(
                f"Rotation #{result['rotation']} - "
                f"Euler({euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°): "
            )
            f.write(f"Defocus = {result['description']}\n")

    print(f"\nAnalysis results written to {output_file}")


def main() -> None:
    """Main function to calculate the center vector between two PDB structures."""
    # Setup argparse
    args = setup_argparse()

    # Calculate relative position and orientation vectors
    vector_data = calculate_relative_vectors(args.pdb_file1, args.pdb_file2)

    # Generate random rotation samples
    rotations_tensor = generate_rotation_samples(args.num_rotations)

    # Process each rotation and collect results
    defocus_results = process_rotations(vector_data["vector"], rotations_tensor)

    # Write results to file
    write_results_to_file(
        args.output_file, args.pdb_file1, args.pdb_file2, vector_data, defocus_results
    )


if __name__ == "__main__":
    main()
