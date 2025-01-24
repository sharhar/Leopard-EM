"""Root-level model for serialization and validation of 2DTM parameters."""

import os
from typing import Any, ClassVar

import torch
from pydantic import ConfigDict, field_validator
from torch_fourier_slice.dft_utils import fftshift_3d

from tt2dtm.backend import core_match_template
from tt2dtm.models.computational_config import ComputationalConfig
from tt2dtm.models.correlation_filters import PreprocessingFilters
from tt2dtm.models.defocus_search_config import DefocusSearchConfig
from tt2dtm.models.match_template_result import MatchTemplateResult
from tt2dtm.models.optics_group import OpticsGroup
from tt2dtm.models.orientation_search_config import OrientationSearchConfig
from tt2dtm.models.pixel_size_search_config import PixelSizeSearchConfig
from tt2dtm.models.types import BaseModel2DTM, ExcludedTensor
from tt2dtm.utils.data_io import load_mrc_image, load_mrc_volume
from tt2dtm.utils.pre_processing import (
    calculate_ctf_filter_stack,
    calculate_searched_orientations,
    calculate_whitening_filter_template,
    do_image_preprocessing,
)


def select_gpu_devices(gpu_ids: int | list[int]) -> list[torch.device]:
    """Convert requested GPU IDs to torch device objects.

    Parameters
    ----------
    gpu_ids : int | list[int]
        GPU ID(s) to use for computation.

    Returns
    -------
    list[torch.device]
    """
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]

    devices = [torch.device(f"cuda:{gpu_id}") for gpu_id in gpu_ids]

    return devices


class MatchTemplateManager(BaseModel2DTM):
    """Model holding parameters necessary for running full orientation 2DTM.

    Attributes
    ----------
    micrograph_path : str
        Path to the micrograph .mrc file.
    template_volume_path : str
        Path to the template volume .mrc file.
    micrograph : ExcludedTensor
        Image to run template matching on. Not serialized.
    template_volume : ExcludedTensor
        Template volume to match against. Not serialized.
    optics_group : OpticsGroup
        Optics group parameters for the imaging system on the microscope.
    defocus_search_config : DefocusSearchConfig
        Parameters for searching over defocus values.
    pixel_size_search_config : PixelSizeSearchConfig
        Parameters for searching over pixel sizes.
    orientation_search_config : OrientationSearchConfig
        Parameters for searching over orientation angles.
    preprocessing_filters : PreprocessingFilters
        Configurations for the preprocessing filters to apply during
        correlation.
    match_template_result : MatchTemplateResult
        Result of the match template program stored as an instance of the
        `MatchTemplateResult` class.
    computational_config : ComputationalConfig
        Parameters for controlling computational resources.

    Methods
    -------
    apply_preprocessing_filters()
        TODO: Implement this method.
    run_match_template()
        TODO: Implement this method.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    # Serialized attributes
    micrograph_path: str
    template_volume_path: str
    optics_group: OpticsGroup
    defocus_search_config: DefocusSearchConfig
    orientation_search_config: OrientationSearchConfig
    pixel_size_search_config: PixelSizeSearchConfig
    preprocessing_filters: PreprocessingFilters
    match_template_result: MatchTemplateResult
    computational_config: ComputationalConfig

    # Non-serialized large array-like attributes
    micrograph: ExcludedTensor
    template_volume: ExcludedTensor

    ###########################
    ### Pydantic Validators ###
    ###########################

    @field_validator("micrograph_path")  # type: ignore
    def validate_micrograph_path(cls, v) -> str:
        """Ensure the micrograph file exists."""
        if not os.path.exists(v):
            raise ValueError(f"File '{v}' for micrograph does not exist.")

        return str(v)

    @field_validator("template_volume_path")  # type: ignore
    def validate_template_volume_path(cls, v) -> str:
        """Ensure the template volume file exists."""
        if not os.path.exists(v):
            raise ValueError(f"File '{v}' for template volume does not exist.")

        return str(v)

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Load the data from the MRC files
        self.micrograph = load_mrc_image(self.micrograph_path)
        self.template_volume = load_mrc_volume(self.template_volume_path)

    ############################################
    ### Functional (data processing) methods ###
    ############################################

    def make_backend_core_function_kwargs(self) -> dict[str, Any]:
        """Generates the keyword arguments for backend call from held parameters."""
        image = self.micrograph
        template = self.template_volume
        image_shape = image.shape
        template_shape = template.shape[-2:]

        whitening_filter = calculate_whitening_filter_template(image, template.shape)
        image_preprocessed_dft = do_image_preprocessing(image)

        defocus_values = self.defocus_search_config.defocus_values
        defocus_values = torch.tensor(defocus_values, dtype=torch.float32)
        ctf_filters = calculate_ctf_filter_stack(
            pixel_size=self.optics_group.pixel_size,
            template_shape=template_shape,
            defocus_u=self.defocus_search_config.defocus_u * 1e-4,  # A to um
            defocus_v=self.defocus_search_config.defocus_v * 1e-4,  # A to um
            astigmatism_angle=self.defocus_search_config.defocus_astigmatism_angle,
            defocus_min=self.defocus_search_config.defocus_min * 1e-4,  # A to um
            defocus_max=self.defocus_search_config.defocus_max * 1e-4,  # A to um
            defocus_step=self.defocus_search_config.defocus_step * 1e-4,  # A to um
            amplitude_contrast_ratio=self.optics_group.amplitude_contrast,
            spherical_aberration=self.optics_group.spherical_aberration,
            phase_shift=self.optics_group.phase_shift,
            voltage=self.optics_group.voltage,
            ctf_B_factor=self.optics_group.ctf_B_factor,
        )

        orientations = calculate_searched_orientations(
            in_plane_angular_step=self.orientation_search_config.in_plane_angular_step,
            out_of_plane_angular_step=self.orientation_search_config.out_of_plane_angular_step,
            phi_min=self.orientation_search_config.phi_min,
            phi_max=self.orientation_search_config.phi_max,
            theta_min=self.orientation_search_config.theta_min,
            theta_max=self.orientation_search_config.theta_max,
            psi_min=self.orientation_search_config.psi_min,
            psi_max=self.orientation_search_config.psi_max,
            template_symmetry=self.orientation_search_config.template_symmetry,
        )
        orientations = orientations.to(torch.float32)

        device_list = select_gpu_devices(self.computational_config.gpu_ids)

        template_dft = fftshift_3d(template, rfft=False)
        template_dft = torch.fft.rfftn(template_dft, dim=(-3, -2, -1))
        template_dft = fftshift_3d(template_dft, rfft=True)

        return {
            "image_dft": image_preprocessed_dft,
            "template_dft": template_dft,
            "ctf_filters": ctf_filters,
            "whitening_filter_template": whitening_filter,
            "defocus_values": defocus_values,
            "orientations": orientations,
            "image_shape": image_shape,
            "template_shape": template_shape,
            "device": device_list,
        }

    def run_match_template(self, projection_batch_size: int = 1) -> None:
        """Runs the base match template in pytorch.

        NOTE: not yet implemented.
        """
        core_kwargs = self.make_backend_core_function_kwargs()
        results = core_match_template(**core_kwargs)

        # TODO: Place results into the `MatchTemplateResult` object and save it.
        _ = results
