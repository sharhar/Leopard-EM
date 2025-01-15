"""Reading, storing, and exporting results from the match_template program."""

import json
import os
from pathlib import Path
from typing import Annotated

import torch
import yaml  # type: ignore
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from tt2dtm.utils.data_io import load_mrc_image

# Pydantic type-hint to exclude tensor from JSON schema/dump (still attribute)
ExcludedTensor = Annotated[torch.Tensor, Field(default=None, exclude=True)]


def check_file_path_and_permissions(path: str, allow_overwrite: bool) -> None:
    """Ensures path is writable and it does not exist, if `allow_overwrite` is False."""
    # 1. Create path to file, if it does not exist
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # 2. Check write permissions
    if not os.access(directory, os.W_OK):
        raise ValueError(
            f"Directory '{directory}' does not permit writing."
            f"Will be unable to write results to '{path}'."
        )

    # 3. Check if file exists
    if not allow_overwrite and os.path.exists(path):
        raise ValueError(
            f"File '{path}' already exists, but 'allow_file_overwrite' "
            "is False. Set 'allow_file_overwrite' to True to permit. "
            "overwriting.\n"
            "WARNING: Overwriting will delete the existing file(s)!"
        )


class MatchTemplateResult(BaseModel):
    """Class to hold and export results from the match_template program.

    Attributes
    ----------
    allow_file_overwrite : bool = False
        Weather to allow overwriting of existing files. Default is False.
        WARNING: Setting to True can overwrite existing files!
    mip_path : str | Path
        Path to the output maximum intensity projection (MIP) file.
    scaled_mip_path : str | Path
        Path to the output scaled MIP file.
    correlation_average_path : str | Path
        Path to the output correlation average file.
    correlation_variance_path : str | Path
        Path to the output correlation variance file.
    orientation_psi_path : str | Path
        Path to the output orientation psi file.
    orientation_theta_path : str | Path
        Path to the output orientation theta file.
    orientation_phi_path : str | Path
        Path to the output orientation phi file.
    relative_defocus_path : str | Path
        Path to the output relative defocus file.
    mip : torch.Tensor
        Maximum intensity projection (MIP).
    scaled_mip : torch.Tensor
        Scaled MIP.
    correlation_average : torch.Tensor
        Correlation average.
    correlation_variance : torch.Tensor
        Correlation variance.
    orientation_psi : torch.Tensor
        Best orientation angle psi.
    orientation_theta : torch.Tensor
        Best orientation angle theta.
    orientation_phi : torch.Tensor
        Best orientation angle phi.
    relative_defocus : torch.Tensor
        Best relative defocus.

    Methods
    -------
    TODO: annotate methods
    """

    # TODO: Implement compression options.

    # Serialized attributes
    # NOTE: This overwrite attribute is a bit overbearing currently. I predict
    # it will lead to headaches when attempting to load a result, this is set
    # to True, and the result files already exist.
    # TODO: Figure how to handle data overwrite prevention (and file write
    # perms) before running expensive GPU computations.
    allow_file_overwrite: bool = False
    mip_path: str
    scaled_mip_path: str
    correlation_average_path: str
    correlation_variance_path: str
    orientation_psi_path: str
    orientation_theta_path: str
    orientation_phi_path: str
    relative_defocus_path: str

    # Large array-like attributes saved to individual files (not in JSON)
    mip: ExcludedTensor
    scaled_mip: ExcludedTensor
    correlation_average: ExcludedTensor
    correlation_variance: ExcludedTensor
    orientation_psi: ExcludedTensor
    orientation_theta: ExcludedTensor
    orientation_phi: ExcludedTensor
    relative_defocus: ExcludedTensor

    ###########################
    ### Pydantic Validators ###
    ###########################

    @model_validator  # type: ignore
    def validate_paths(self) -> Self:
        """Validate output paths for write permissions and overwriting.

        Note: This method runs after instantiation, so attributes are already
        set. We can safely access them with `self`.

        Returns
        -------
        Self
            The validated instance.

        Raises
        ------
        ValueError
            If the output paths are not writable or do not permit overwriting.
        """
        # 1. Check write permissions and overwriting for each path
        paths = [
            self.mip_path,
            self.scaled_mip_path,
            self.correlation_average_path,
            self.correlation_variance_path,
            self.orientation_psi_path,
            self.orientation_theta_path,
            self.orientation_phi_path,
            self.relative_defocus_path,
        ]
        for path in paths:
            check_file_path_and_permissions(path, self.allow_file_overwrite)

        return self

    #######################################
    ### Class methods for instantiation ###
    #######################################

    @classmethod
    def from_json(cls, json_path: str | Path) -> "MatchTemplateResult":
        """Load a MatchTemplateResult from a serialized JSON file."""
        with open(json_path) as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MatchTemplateResult":
        """Load a MatchTemplateResult from a serialized YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    ############################################
    ### Functional (data processing) methods ###
    ############################################

    def load_tensors_from_paths(self) -> None:
        """Use the held paths to load tensors into memory.

        NOTE: Currently only supports .mrc files.
        """
        self.mip = load_mrc_image(self.mip_path)
        self.scaled_mip = load_mrc_image(self.scaled_mip_path)
        self.correlation_average = load_mrc_image(self.correlation_average_path)
        self.correlation_variance = load_mrc_image(self.correlation_variance_path)
        self.orientation_psi = load_mrc_image(self.orientation_psi_path)
        self.orientation_theta = load_mrc_image(self.orientation_theta_path)
        self.orientation_phi = load_mrc_image(self.orientation_phi_path)
        self.relative_defocus = load_mrc_image(self.relative_defocus_path)

    ######################
    ### Export methods ###
    ######################

    def export_results(self) -> None:
        """Export the torch.Tensor results to the specified mrc files."""
        # TODO: Handle pixel_size and other mrc metadata when exporting
        raise NotImplementedError
