"""Set of classes for configuring correlation filters in 2DTM."""

from typing import Annotated, Optional

import torch
from pydantic import Field
from torch_fourier_filter.bandpass import bandpass_filter
from torch_fourier_filter.dft_utils import fftfreq_grid
from torch_fourier_filter.phase_randomize import phase_randomize
from torch_fourier_filter.utils import curve_1dim_to_ndim
from torch_fourier_filter.whitening import whitening_filter

from tt2dtm.pydantic_models.types import BaseModel2DTM


class WhiteningFilterConfig(BaseModel2DTM):
    """Configuration for the whitening filter.

    Attributes
    ----------
    enabled : bool
        If True, apply a whitening filter to the input image and template projections.
        Default is True.
    num_freq_bins : Optional[int]
        Number of frequency bins (in 1D) to use when calculating the power spectrum.
        Default is None which automatically determines the number of bins based on the
        input image size.
    max_freq : Optional[float]
        Maximum frequency, in terms of Nyquist frequency, to use when calculating the
        whitening filter. Default is 0.5 with values pixels above 0.5 being set to 1.0
        in the filter (i.e. no frequency scaling).
    do_power_spectrum : Optional[bool]
        If True, calculate the power spectral density from the power of the
        input image. Default is True. If False, then the power spectral density is
        calculated from the amplitude of the input image.

    Methods
    -------
    calculate_whitening_filter(ref_img_rfft, output_shape, output_rfft, output_fftshift)
        Helper function for the whitening filter based on the input reference image and
        held configuration parameters.
    """

    enabled: bool = True
    num_freq_bins: Optional[int] = None
    max_freq: Optional[float] = 0.5  # in terms of Nyquist frequency
    do_power_spectrum: Optional[bool] = True

    def calculate_whitening_filter(
        self,
        ref_img_rfft: torch.Tensor,
        output_shape: Optional[tuple[int, ...]] = None,
        output_rfft: bool = True,
        output_fftshift: bool = False,
    ) -> torch.Tensor:
        """Helper function for the whitening filter based on the input reference image.

        NOTE: This function is a wrapper around the `whitening_filter` function from
        the `torch_fourier_filter` package. It expects the input image to be RFFT'd
        and unshifted (zero-frequency component at the top-left corner). The output
        can be of any shape, but the default is to return a filer of the same input
        shape.

        Parameters
        ----------
        ref_img_rfft : torch.Tensor
            The reference image (RFFT'd and unshifted) to calculate the whitening
            filter from.
        output_shape : Optional[tuple[int, ...]]
            Desired output shape of the whitening filter. This is the filter shape in
            Fourier space *not* real space (like in the torch_fourier_filter package).
            Default is None, which is the same as the input shape.
        output_rfft : Optional[bool]
            If True, filter corresponds to a Fourier transform using the RFFT.
            Default is None, which is the same as the 'rfft' parameter.
        output_fftshift : Optional[bool]
            If True, filter corresponds to a Fourier transform followed
            by an fftshift. Default is None, which is the same as the 'fftshift'
            parameter.

        Returns
        -------
        torch.Tensor
            The whitening filter with frequencies calculated from the input reference
            image.
        """
        if output_shape is None:
            output_shape = ref_img_rfft.shape

        # Handle case where whitening filter is disabled
        if not self.enabled:
            return torch.ones(output_shape, dtype=ref_img_rfft.dtype)

        # Convert to real-space shape for function call
        output_shape = output_shape[:-1] + (2 * (output_shape[-1] - 1),)

        return whitening_filter(
            image_dft=ref_img_rfft,
            rfft=True,
            fftshift=False,
            dim=(-2, -1),
            num_freq_bins=self.num_freq_bins,
            max_freq=self.max_freq,
            do_power_spectrum=self.do_power_spectrum,
            output_shape=output_shape,
            output_rfft=output_rfft,
            output_fftshift=output_fftshift,
        )


class PhaseRandomizationFilterConfig(BaseModel2DTM):
    """Configuration for phase randomization filter.

    NOTE: Something is not working with the underlying torch_fourier_filter code
    for phase randomization.

    Attributes
    ----------
    enabled : bool
        If True, apply a phase randomization filter to the input image. Default
        is False.
    cuton : float
        Spatial resolution, in terms of Nyquist, above which to randomize the phase.

    Methods
    -------
    calculate_phase_randomization_filter(ref_img_rfft)
        Helper function for the phase randomization filter based on the input reference
        image and held configuration parameters.
    """

    enabled: bool = False
    cuton: Optional[Annotated[float, Field(ge=0.0)]] = None

    def calculate_phase_randomization_filter(
        self, ref_img_rfft: torch.Tensor
    ) -> torch.Tensor:
        """Helper function for phase randomization filter based on the reference image.

        Parameters
        ----------
        ref_img_rfft : torch.Tensor
            The reference image to use as a template for phase randomization. This
            should be RFFT'd and unshifted (zero-frequency component at the top-left
            corner).
        """
        output_shape = ref_img_rfft.shape

        # Handle case where phase randomization filter is disabled
        if not self.enabled:
            return torch.ones(output_shape, dtype=ref_img_rfft.dtype)

        # Fix for underlying shape bug in torch_fourier_filter
        # TODO: fix this in package
        output_shape = output_shape[:-1] + (2 * (output_shape[-1] - 1),)

        return phase_randomize(
            dft=ref_img_rfft,
            image_shape=output_shape,
            rfft=True,
            fftshift=False,
            cuton=self.cuton,
        )


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

    enabled: bool = False
    low_pass: Optional[Annotated[float, Field(ge=0.0)]] = None
    high_pass: Optional[Annotated[float, Field(ge=0.0)]] = None
    falloff: Optional[Annotated[float, Field(ge=0.0)]] = None

    def calculate_bandpass_filter(self, output_shape: tuple[int, ...]) -> torch.Tensor:
        """Helper function for bandpass filter based on the desired output shape.

        Note that the output will be in terms of an RFFT'd and unshifted (zero-frequency
        component at the top-left corner) image.

        Parameters
        ----------
        output_shape : tuple[int, ...]
            Desired output shape of the bandpass filter in Fourier space. This is the
            filter shape in Fourier space *not* real space (like in the
            torch_fourier_filter package).

        Returns
        -------
        torch.Tensor
            The bandpass filter for the desired output shape.
        """
        # Handle case where bandpass filter is disabled
        if not self.enabled:
            return torch.ones(output_shape, dtype=torch.float32)

        # Account for None values
        low = self.low_pass if self.low_pass is not None else 0.0
        high = self.high_pass if self.high_pass is not None else 1.0
        falloff = self.falloff if self.falloff is not None else 0.0

        # Convert to real-space shape for function call
        output_shape = output_shape[:-1] + (2 * (output_shape[-1] - 1),)

        return bandpass_filter(
            low=low,
            high=high,
            falloff=falloff,
            image_shape=output_shape,
            rfft=True,
            fftshift=False,
        )


class ArbitraryCurveFilterConfig(BaseModel2DTM):
    """Class holding frequency and amplitude values for arbitrary curve filter.

    Attributes
    ----------
    frequencies : list[float]
        List of spatial frequencies (in terms of Nyquist) for the corresponding
        amplitudes.
    amplitudes : list[float]
        List of amplitudes for the corresponding spatial frequencies.
    """

    enabled: bool = False
    frequencies: Optional[list[float]] = None  # in terms of Nyquist frequency
    amplitudes: Optional[list[float]] = None

    def calculate_arbitrary_curve_filter(
        self, output_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Calculates the curve filter for the desired output shape.

        Parameters
        ----------
        output_shape : tuple[int, ...]
            Desired output shape of the curve filter in Fourier space. This is the
            filter shape in Fourier space *not* real space (like in the
            torch_fourier_filter package).

        Returns
        -------
        torch.Tensor
            The curve filter for the desired output shape.
        """
        if not self.enabled:
            return torch.ones(output_shape, dtype=torch.float32)

        # Ensure that neither frequencies nor amplitudes are None
        if self.frequencies is None or self.amplitudes is None:
            raise ValueError(
                "When enabled, both 'frequencies' and 'amplitudes' must be provided."
            )

        # Convert to real-space shape for function call
        output_shape = output_shape[:-1] + (2 * (output_shape[-1] - 1),)

        freq_grid = fftfreq_grid(
            image_shape=output_shape,
            rfft=True,
            fftshift=False,
            norm=True,
        )

        frequencies = torch.tensor(self.frequencies)
        amplitudes = torch.tensor(self.amplitudes)
        filter_ndim = curve_1dim_to_ndim(
            frequency_1d=frequencies,
            values_1d=amplitudes,
            frequency_grid=freq_grid,
            fill_lower=1.0,  # Fill oob areas with ones (no scaling)
            fill_upper=1.0,
        )

        return filter_ndim


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

    whitening_filter: WhiteningFilterConfig = WhiteningFilterConfig()
    bandpass_filter: BandpassFilterConfig = BandpassFilterConfig()
    phase_randomization_filter: PhaseRandomizationFilterConfig = (
        PhaseRandomizationFilterConfig()
    )
    arbitrary_curve_filter: ArbitraryCurveFilterConfig = ArbitraryCurveFilterConfig()
