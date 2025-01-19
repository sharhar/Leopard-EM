"""Set of classes for configuring correlation filters in 2DTM."""

from typing import Annotated, Optional

from pydantic import Field

from tt2dtm.models.types import BaseModel2DTM


class PhaseRandomizationFilterConfig(BaseModel2DTM):
    """Configuration for phase randomization filter.

    Attributes
    ----------
    enabled : bool
        If True, apply a phase randomization filter to the input image. Default
        is False.
    cuton : float
        Spatial resolution, in Angstroms, above which to randomize the phase.
    """

    enabled: Annotated[bool, Field(...)] = False
    cuton: Optional[Annotated[float, Field(ge=0.0)]] = None


class WhiteningFilterConfig(BaseModel2DTM):
    """Configuration for the whitening filter.

    Attributes
    ----------
    enabled : bool
        If True, apply a whitening filter to the input image. Default is True.
    power_spectrum : bool
        If True, calculate the whitening filter from the power spectrum. If
        False, calculate the whitening filter from the amplitude spectrum.
        Default is True.
    smoothing : float
        Smoothing factor for the whitening filter. Default is 0.0, which is no
        smoothing.
    """

    enabled: Annotated[bool, Field(...)] = True
    power_spectrum: Annotated[bool, Field(...)] = True
    smoothing: Annotated[float, Field(ge=0.0)] = 0.0


class BandpassFilterConfig(BaseModel2DTM):
    """Configuration for the bandpass filter.

    Attributes
    ----------
    enabled : bool
        If True, apply a bandpass filter to correlation during template
        matching. Default is False.
    low_pass : Optional[float]
        Low pass filter cutoff frequency. Default is None, which is no low
        pass filter.
    high_pass : Optional[float]
        High pass filter cutoff frequency. Default is None, which is no high
        pass filter.
    falloff : Optional[float]
        Falloff factor for bandpass filter. Default is 0.0, which is no
        falloff.
    """

    enabled: Annotated[bool, Field(...)] = False
    low_pass: Optional[Annotated[float, Field(ge=0.0)]] = None
    high_pass: Optional[Annotated[float, Field(ge=0.0)]] = None
    falloff: Optional[Annotated[float, Field(ge=0.0)]] = None


class PreprocessingFilters(BaseModel2DTM):
    """Configuration class for all preprocessing filters.

    Attributes
    ----------
    whitening_filter_config : WhiteningFilterConfig
        Configuration for the whitening filter.
    bandpass_filter_config : BandpassFilterConfig
        Configuration for the bandpass filter.
    phase_randomization_filter_config : PhaseRandomizationFilterConfig
        Configuration for the phase randomization filter.
    """

    whitening_filter_config: WhiteningFilterConfig
    bandpass_filter_config: BandpassFilterConfig
    phase_randomization_filter_config: PhaseRandomizationFilterConfig
