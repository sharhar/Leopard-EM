"""Root-level model for serialization and validation of 2DTM parameters."""

import os
from typing import Any, ClassVar

from pydantic import ConfigDict, field_validator

from tt2dtm.models.computational_config import ComputationalConfig
from tt2dtm.models.correlation_filters import PreprocessingFilters
from tt2dtm.models.defocus_search_config import DefocusSearchConfig
from tt2dtm.models.match_template_result import MatchTemplateResult
from tt2dtm.models.optics_group import OpticsGroup
from tt2dtm.models.orientation_search_config import OrientationSearchConfig
from tt2dtm.models.pixel_size_search_config import PixelSizeSearchConfig
from tt2dtm.models.types import BaseModel2DTM, ExcludedTensor
from tt2dtm.utils.data_io import load_mrc_image, load_mrc_volume


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

    def apply_preprocessing_filters(self) -> None:
        """Calculates and applies necessary filters for template matching.

        NOTE: not yet implemented.
        """
        raise NotImplementedError

    def run_match_template(self) -> None:
        """Runs the base match template in pytorch.

        NOTE: not yet implemented.
        """
        self.apply_preprocessing_filters()
        raise NotImplementedError
