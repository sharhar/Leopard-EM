"""Pydantic model for running the refine template program."""

from typing import Any, ClassVar

import pandas as pd
from pydantic import ConfigDict, field_validator

from tt2dtm.pydantic_models.computational_config import ComputationalConfig
from tt2dtm.pydantic_models.correlation_filters import PreprocessingFilters
from tt2dtm.pydantic_models.defocus_search import DefocusSearchConfig
from tt2dtm.pydantic_models.match_template_manager import MatchTemplateManager
from tt2dtm.pydantic_models.orientation_search import RefineOrientationConfig
from tt2dtm.pydantic_models.particle_stack import ParticleStack
from tt2dtm.pydantic_models.refine_template_result import RefineTemplateResult
from tt2dtm.pydantic_models.types import BaseModel2DTM, ExcludedTensor
from tt2dtm.utils.data_io import load_mrc_volume


class RefineTemplateManager(BaseModel2DTM):
    """Model holding parameters necessary for running the refine template program.

    Attributes
    ----------
    TODO: Fill in attributes.

    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    template_volume_path: str  # In df per-particle, but ensure only one reference
    particle_stack: ParticleStack
    defocus_refinement_config: DefocusSearchConfig
    orientation_refinement_config: RefineOrientationConfig
    preprocessing_filters: PreprocessingFilters
    refine_template_result: RefineTemplateResult
    computational_config: ComputationalConfig

    # Excluded tensors
    template_volume: ExcludedTensor

    @classmethod
    def from_results_csv(cls, results_csv_path: str) -> "RefineTemplateManager":
        """Take tabular data (from csv) and create a RefineTemplateManager object.

        Parameters
        ----------
        results_csv_path : str
            Path to the CSV file containing the per-particle data.

        Returns
        -------
        RefineTemplateManager
        """
        df = pd.read_csv(results_csv_path)

        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> "RefineTemplateManager":
        """Tabular data (from DataFrame) and create a RefineTemplateManager object."""
        # # Ensure there is only one template volume path
        # template_volume_path = dataframe["template_volume_path"].unique()
        # if len(template_volume_path) != 1:
        #     raise ValueError("Multiple template volume paths found in dataframe.")
        # template_volume_path = template_volume_path[0]

        # # Tensor data
        # pos_x = dataframe["pos_x"].values
        # pos_y = dataframe["pos_y"].values
        # mip = dataframe["mip"].values
        # scaled_mip = dataframe["scaled_mip"].values
        # psi = dataframe["psi"].values
        # theta = dataframe["theta"].values
        # phi = dataframe["phi"].values
        # defocus = dataframe["defocus"].values
        # corr_avg = dataframe["corr_average"].values
        # corr_var = dataframe["corr_variance"].values

        # micrograph_paths = dataframe["micrograph_path"].values
        # micrograph_paths_unique = micrograph_paths  # sorted
        # template_volume_path = dataframe["template_volume_path"].values

        raise NotImplementedError("Method not implemented yet.")

    @classmethod
    def from_match_template_manager(
        cls, mt_manager: MatchTemplateManager
    ) -> "RefineTemplateManager":
        """Creates a RefineTemplateManager object from a MatchTemplateManager object."""
        raise NotImplementedError("Method not implemented yet.")

    def __init__(self, skip_mrc_preloads: bool = False, **data: Any):
        super().__init__(**data)

        # Load the data from the MRC files
        if not skip_mrc_preloads:
            self.template_volume = load_mrc_volume(self.template_volume_path)

    def load_particle_stack(self) -> None:
        """Extracts particles from known positions in micrographs."""
        raise NotImplementedError("Method not implemented yet.")

    @field_validator("micrograph_paths")  # type: ignore
    def validate_micrograph_paths(cls, value: str | list[str]) -> list[str]:
        """Validate the micrograph paths."""
        if isinstance(value, str):
            return [value]
        return value

    def particles_to_dataframe(self) -> pd.DataFrame:
        """Export refined particles as dataframe."""
        raise NotImplementedError("Method not implemented yet.")

    def results_to_dataframe(self) -> pd.DataFrame:
        """Export full results as dataframe."""
        raise NotImplementedError("Method not implemented yet.")

    def particle_stack_to_dataframe(self) -> pd.DataFrame:
        """Export particle stack information as dataframe."""
        raise NotImplementedError("Method not implemented yet.")

    def save_particle_stack(self, save_dir: str) -> None:
        """Save the particle stack to disk."""
        raise NotImplementedError("Method not implemented yet.")
