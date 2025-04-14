"""Pydantic model for running the refine template program."""

from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import ConfigDict

from leopard_em.backend.core_refine_template import core_refine_template
from leopard_em.pydantic_models.computational_config import ComputationalConfig
from leopard_em.pydantic_models.correlation_filters import PreprocessingFilters
from leopard_em.pydantic_models.custom_types import BaseModel2DTM, ExcludedTensor
from leopard_em.pydantic_models.defocus_search import DefocusSearchConfig
from leopard_em.pydantic_models.formats import REFINED_DF_COLUMN_ORDER
from leopard_em.pydantic_models.orientation_search import RefineOrientationConfig
from leopard_em.pydantic_models.particle_stack import ParticleStack
from leopard_em.pydantic_models.pixel_size_search import PixelSizeSearchConfig
from leopard_em.pydantic_models.utils import (
    _setup_ctf_kwargs_from_particle_stack,
    preprocess_image,
    volume_to_rfft_fourier_slice,
)
from leopard_em.utils.data_io import load_mrc_volume


class RefineTemplateManager(BaseModel2DTM):
    """Model holding parameters necessary for running the refine template program.

    Attributes
    ----------
    template_volume_path : str
        Path to the template volume MRC file.
    particle_stack : ParticleStack
        Particle stack object containing particle data.
    defocus_refinement_config : DefocusSearchConfig
        Configuration for defocus refinement.
    pixel_size_refinement_config : PixelSizeSearchConfig
        Configuration for pixel size refinement.
    orientation_refinement_config : RefineOrientationConfig
        Configuration for orientation refinement.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    computational_config : ComputationalConfig
        What computational resources to allocate for the program.
    template_volume : ExcludedTensor
        The template volume tensor (excluded from serialization).

    Methods
    -------
    TODO serialization/import methods
    __init__(self, skip_mrc_preloads: bool = False, **data: Any)
        Initialize the refine template manager.
    make_backend_core_function_kwargs(self) -> dict[str, Any]
        Create the kwargs for the backend refine_template core function.
    run_refine_template(self, orientation_batch_size: int = 64) -> None
        Run the refine template program.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    template_volume_path: str  # In df per-particle, but ensure only one reference
    particle_stack: ParticleStack
    defocus_refinement_config: DefocusSearchConfig
    pixel_size_refinement_config: PixelSizeSearchConfig
    orientation_refinement_config: RefineOrientationConfig
    preprocessing_filters: PreprocessingFilters
    computational_config: ComputationalConfig

    # Excluded tensors
    template_volume: ExcludedTensor

    def __init__(self, skip_mrc_preloads: bool = False, **data: Any):
        super().__init__(**data)

        # Load the data from the MRC files
        if not skip_mrc_preloads:
            self.template_volume = load_mrc_volume(self.template_volume_path)

    def make_backend_core_function_kwargs(
        self, prefer_refined_angles: bool = False
    ) -> dict[str, Any]:
        """Create the kwargs for the backend refine_template core function.

        Parameters
        ----------
        prefer_refined_angles : bool
            Whether to use the refined angles from the particle stack. Defaults to
            False.
        """
        device_list = self.computational_config.gpu_devices

        # Ensure the template is loaded in as a Tensor object
        if self.template_volume is None:
            self.template_volume = load_mrc_volume(self.template_volume_path)
        if not isinstance(self.template_volume, torch.Tensor):
            template = torch.from_numpy(self.template_volume)
        else:
            template = self.template_volume

        # Extract out the regions of interest (particles) based on the particle stack
        particle_images = self.particle_stack.construct_image_stack(
            pos_reference="center",
            padding_value=0.0,
            handle_bounds="pad",
            padding_mode="constant",
        )
        # pylint: disable=E1102
        particle_images_dft = torch.fft.rfftn(particle_images, dim=(-2, -1))
        particle_images_dft[..., 0, 0] = 0.0 + 0.0j  # Zero out DC component

        bandpass_filter = (
            self.preprocessing_filters.bandpass_filter.calculate_bandpass_filter(
                particle_images_dft.shape[-2:]
            )
        )

        # Calculate and apply the filters for the particle image stack
        filter_stack = self.particle_stack.construct_filter_stack(
            self.preprocessing_filters, output_shape=particle_images_dft.shape[-2:]
        )

        particle_images_dft = preprocess_image(
            image_rfft=particle_images_dft,
            cumulative_fourier_filters=filter_stack,
            bandpass_filter=bandpass_filter,
        )

        # Calculate the filters applied to each template (besides CTF)
        projective_filters = self.particle_stack.construct_filter_stack(
            self.preprocessing_filters,
            output_shape=(template.shape[-2], template.shape[-1] // 2 + 1),
        )

        template_dft = volume_to_rfft_fourier_slice(template)

        # The set of "best" euler angles from match template search
        # Check if refined angles exist, otherwise use the original angles
        euler_angles = self.particle_stack.get_euler_angles(prefer_refined_angles)

        # The relative Euler angle offsets to search over
        euler_angle_offsets = self.orientation_refinement_config.euler_angles_offsets

        # The best defocus values for each particle (+ astigmatism)
        defocus_u = self.particle_stack.absolute_defocus_u
        defocus_v = self.particle_stack.absolute_defocus_v
        defocus_angle = torch.tensor(self.particle_stack["astigmatism_angle"])

        # The relative defocus values to search over
        defocus_offsets = self.defocus_refinement_config.defocus_values

        # The relative pixel size values to search over
        pixel_size_offsets = self.pixel_size_refinement_config.pixel_size_values

        ctf_kwargs = _setup_ctf_kwargs_from_particle_stack(
            self.particle_stack, (template.shape[-2], template.shape[-1])
        )

        return {
            "particle_stack_dft": particle_images_dft,
            "template_dft": template_dft,
            "euler_angles": euler_angles,
            "euler_angle_offsets": euler_angle_offsets,
            "defocus_u": defocus_u,
            "defocus_v": defocus_v,
            "defocus_angle": defocus_angle,
            "defocus_offsets": defocus_offsets,
            "pixel_size_offsets": pixel_size_offsets,
            "ctf_kwargs": ctf_kwargs,
            "projective_filters": projective_filters,
            "device": device_list,  # Pass all devices to core_refine_template
        }

    def run_refine_template(
        self, output_dataframe_path: str, orientation_batch_size: int = 64
    ) -> None:
        """Run the refine template program and saves the resultant DataFrame to csv.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the refined particle data.
        orientation_batch_size : int
            Number of orientations to process at once. Defaults to 64.
        """
        backend_kwargs = self.make_backend_core_function_kwargs()

        result = self.get_refine_result(backend_kwargs, orientation_batch_size)

        self.refine_result_to_dataframe(
            output_dataframe_path=output_dataframe_path, result=result
        )

    def get_refine_result(
        self, backend_kwargs: dict, orientation_batch_size: int = 64
    ) -> dict[str, np.ndarray]:
        """Get refine template result.

        Parameters
        ----------
        backend_kwargs : dict
            Keyword arguments for the backend processing
        orientation_batch_size : int
            Number of orientations to process at once. Defaults to 64.

        Returns
        -------
        dict[str, np.ndarray]
            The result of the refine template program.
        """
        # Adjust batch size if orientation search is disabled
        if not self.orientation_refinement_config.enabled:
            orientation_batch_size = 1
        elif (
            self.orientation_refinement_config.euler_angles_offsets.shape[0]
            < orientation_batch_size
        ):
            orientation_batch_size = (
                self.orientation_refinement_config.euler_angles_offsets.shape[0]
            )

        result: dict[str, np.ndarray] = {}
        result = core_refine_template(
            batch_size=orientation_batch_size, **backend_kwargs
        )
        result = {k: v.cpu().numpy() for k, v in result.items()}
        return result

    def refine_result_to_dataframe(
        self, output_dataframe_path: str, result: dict[str, np.ndarray]
    ) -> None:
        """Convert refine template result to dataframe.

        Parameters
        ----------
        output_dataframe_path : str
            Path to save the refined particle data.
        result : dict[str, np.ndarray]
            The result of the refine template program.
        """
        df_refined = self.particle_stack._df.copy()  # pylint: disable=protected-access
        refined_mip = result["refined_cross_correlation"]
        refined_scaled_mip = refined_mip - df_refined["correlation_mean"]
        refined_scaled_mip = refined_scaled_mip / np.sqrt(
            df_refined["correlation_variance"]
        )

        pos_offset_y = result["refined_pos_y"]
        pos_offset_x = result["refined_pos_x"]
        pos_offset_y_ang = pos_offset_y * df_refined["pixel_size"]
        pos_offset_x_ang = pos_offset_x * df_refined["pixel_size"]

        # Add the new columns to the DataFrame
        df_refined["refined_mip"] = refined_mip
        df_refined["refined_scaled_mip"] = refined_scaled_mip

        df_refined["refined_psi"] = result["refined_euler_angles"][:, 2]
        df_refined["refined_theta"] = result["refined_euler_angles"][:, 1]
        df_refined["refined_phi"] = result["refined_euler_angles"][:, 0]

        df_refined["refined_relative_defocus"] = (
            result["refined_defocus_offset"] + df_refined["refined_relative_defocus"]
        )
        df_refined["refined_pixel_size"] = (
            result["refined_pixel_size_offset"] + df_refined["pixel_size"]
        )
        df_refined["refined_pos_y"] = pos_offset_y + df_refined["pos_y"]
        df_refined["refined_pos_x"] = pos_offset_x + df_refined["pos_x"]
        df_refined["refined_pos_y_img"] = pos_offset_y + df_refined["pos_y_img"]
        df_refined["refined_pos_x_img"] = pos_offset_x + df_refined["pos_x_img"]
        df_refined["refined_pos_y_img_angstrom"] = (
            pos_offset_y_ang + df_refined["pos_y_img_angstrom"]
        )
        df_refined["refined_pos_x_img_angstrom"] = (
            pos_offset_x_ang + df_refined["pos_x_img_angstrom"]
        )

        # Reorder the columns
        df_refined = df_refined.reindex(columns=REFINED_DF_COLUMN_ORDER)

        # Save the refined DataFrame to disk
        df_refined.to_csv(output_dataframe_path)
