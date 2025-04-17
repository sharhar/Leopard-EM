"""Utility function associated with the analysis submodule."""

import torch


def filter_peaks_by_distance(
    peak_values: torch.Tensor,
    peak_locations: torch.Tensor,
    distance_threshold: float,
) -> torch.Tensor:
    """Filters peaks in descending order with a distance threshold.

    If any peaks are within a distance threshold of an already picked peak, they are
    suppressed. Precedence is given to peaks with the highest value.

    NOTE: This function operates on 2D data only and distance metrics are measured in
    pixels.

    Parameters
    ----------
    peak_values : torch.Tensor
        Tensor of peak values.
    peak_locations : torch.Tensor
        Tensor of peak locations, shape (N, 2) where N is the number of peaks.
    distance_threshold : float
        Minimum distance between peaks to be considered separate peaks.
    """
    # Sort the peaks in descending order based on their values
    peak_values, sort_indices = torch.sort(peak_values, descending=True)
    peak_locations = peak_locations[sort_indices]

    # Create a boolean mask to record which peaks are taken
    taken_mask = torch.zeros(
        peak_values.size(0), dtype=torch.bool, device=peak_values.device
    )
    picked_peaks = torch.tensor([], dtype=torch.long, device=peak_values.device)

    for i in range(peak_values.size(0)):
        if taken_mask[i]:
            continue

        picked_peaks = torch.cat((picked_peaks, peak_locations[i].unsqueeze(0)), dim=0)

        # Compute distances between current peak and all peaks
        distances = torch.norm(peak_locations - peak_locations[i].float(), dim=1)

        # Mark all peaks closer than distance_threshold as taken
        taken_mask |= distances < distance_threshold

    return picked_peaks
