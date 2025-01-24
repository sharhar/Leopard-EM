"""Reading, storing, and exporting results from the match_template program."""

import os
from typing import ClassVar

from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from tt2dtm.models.types import BaseModel2DTM, ExcludedTensor
from tt2dtm.utils.data_io import load_mrc_image, write_mrc_from_tensor


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


class MatchTemplateResult(BaseModel2DTM):
    """Class to hold and export results from the match_template program.

    TODO: Implement tracking of how far along the template matching is
    (e.g. orientations up to what index have been searched).
    TODO: Implement method for exporting intermediary results in case of error
    or program interruption.
    TODO: Implement functionality for restarting template matching from a
    saved state (e.g. after a program interruption).

    Attributes
    ----------
    allow_file_overwrite : bool = False
        Weather to allow overwriting of existing files. Default is False.
        WARNING: Setting to True can overwrite existing files!
    mip_path : str
        Path to the output maximum intensity projection (MIP) file.
    scaled_mip_path : str
        Path to the output scaled MIP file.
    correlation_average_path : str
        Path to the output correlation average file.
    correlation_variance_path : str
        Path to the output correlation variance file.
    orientation_psi_path : str
        Path to the output orientation psi file.
    orientation_theta_path : str
        Path to the output orientation theta file.
    orientation_phi_path : str
        Path to the output orientation phi file.
    relative_defocus_path : str
        Path to the output relative defocus file.
    pixel_size_path : str
        Path to the output best pixel size file.
    mip : ExcludedTensor
        Maximum intensity projection (MIP).
    scaled_mip : ExcludedTensor
        Scaled MIP.
    correlation_average : ExcludedTensor
        Correlation average.
    correlation_variance : ExcludedTensor
        Correlation variance.
    orientation_psi : ExcludedTensor
        Best orientation angle psi.
    orientation_theta : ExcludedTensor
        Best orientation angle theta.
    orientation_phi : ExcludedTensor
        Best orientation angle phi.
    relative_defocus : ExcludedTensor
        Best relative defocus.
    pixel_size : ExcludedTensor
        Best pixel size.
    total_correlations : int, optional
        Total number of correlations computed. Default is 0, and this field is updated
        automatically after a match_template run.

    Methods
    -------
    TODO: annotate methods
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

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
    pixel_size_path: str

    # Large array-like attributes saved to individual files (not in JSON)
    mip: ExcludedTensor
    scaled_mip: ExcludedTensor
    correlation_average: ExcludedTensor
    correlation_variance: ExcludedTensor
    orientation_psi: ExcludedTensor
    orientation_theta: ExcludedTensor
    orientation_phi: ExcludedTensor
    relative_defocus: ExcludedTensor
    pixel_size: ExcludedTensor
    total_correlations: int = 0

    ###########################
    ### Pydantic Validators ###
    ###########################

    @model_validator(mode="after")  # type: ignore
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
            self.pixel_size_path,
        ]
        for path in paths:
            check_file_path_and_permissions(path, self.allow_file_overwrite)

        return self

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

        paths = [
            self.mip_path,
            self.scaled_mip_path,
            self.correlation_average_path,
            self.correlation_variance_path,
            self.orientation_psi_path,
            self.orientation_theta_path,
            self.orientation_phi_path,
            self.relative_defocus_path,
            self.pixel_size_path,
        ]
        tensors = [
            self.mip,
            self.scaled_mip,
            self.correlation_average,
            self.correlation_variance,
            self.orientation_psi,
            self.orientation_theta,
            self.orientation_phi,
            self.relative_defocus,
            self.pixel_size,
        ]

        for path, tensor in zip(paths, tensors):
            write_mrc_from_tensor(
                data=tensor,
                mrc_path=path,
                mrc_header=None,
                overwrite=self.allow_file_overwrite,
            )
