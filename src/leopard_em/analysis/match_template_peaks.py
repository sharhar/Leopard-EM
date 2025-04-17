"""Data class and helper functions for match template peaks and statistics."""

from typing import NamedTuple

import pandas as pd
import torch


class MatchTemplatePeaks(NamedTuple):
    """Helper class for return value of extract_peaks_and_statistics."""

    pos_y: torch.Tensor
    pos_x: torch.Tensor
    mip: torch.Tensor
    scaled_mip: torch.Tensor
    psi: torch.Tensor
    theta: torch.Tensor
    phi: torch.Tensor
    relative_defocus: torch.Tensor
    correlation_mean: torch.Tensor
    correlation_variance: torch.Tensor
    total_correlations: int


def match_template_peaks_to_dict(peaks: MatchTemplatePeaks) -> dict:
    """Convert MatchTemplatePeaks object to a dictionary."""
    return peaks._asdict()


def match_template_peaks_to_dataframe(peaks: MatchTemplatePeaks) -> pd.DataFrame:
    """Convert MatchTemplatePeaks object to a pandas DataFrame."""
    return pd.DataFrame(peaks._asdict())
