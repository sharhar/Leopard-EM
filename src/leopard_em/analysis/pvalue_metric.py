"""Module for calling true/false positives using the p-value metric."""

import warnings

import numpy as np
import torch
from lmfit import Model, Parameters
from lmfit.model import ModelResult
from scipy import special
from scipy.stats import multivariate_normal

from .match_template_peaks import MatchTemplatePeaks
from .utils import filter_peaks_by_distance


def _params_to_multivariate_normal(
    mu_x: float,
    mu_y: float,
    sigma_x: float,
    sigma_y: float,
    rho: float,
) -> multivariate_normal:
    """Helper function to convert parameters to a multivariate normal distribution."""
    mean = np.array([mu_x, mu_y])
    cov = np.array(
        [
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2],
        ]
    )
    return multivariate_normal(mean=mean, cov=cov)


def probit_transform(x: np.ndarray) -> np.ndarray:
    """Apply the probit transform to the quantiles of input data x.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Transformed array.
    """
    assert x.ndim == 1, "Input array must be 1-dimensional."

    n = x.size

    # Get the sorted indices of the input array
    sorted_indices = np.argsort(x)

    # Create an array of ranks based on the sorted indices
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, n + 1)  # ranks start from 1

    # Calculate the probit transform
    rank_tmp = (ranks - 0.5) / n
    # pylint: disable=no-member
    probit = np.sqrt(2.0) * special.erfinv(2.0 * rank_tmp - 1.0)

    return probit


def fit_full_cov_gaussian_2d(
    data: np.ndarray,
    x_dim: np.ndarray,
    y_dim: np.ndarray,
) -> ModelResult:
    """Fit the full covariance 2D Gaussian to the data using the LMFit package.

    Parameters
    ----------
    data : np.ndarray
        2D array of data to fit.
    x_dim : np.ndarray
        1D array of x-coordinates.
    y_dim : np.ndarray
        1D array of y-coordinates.

    Returns
    -------
    ModelResult
        The result of the fit.
    """
    assert data.ndim == 2, "Data must be a 2D array."
    assert x_dim.ndim == 1, "x_dim must be a 1D array."
    assert y_dim.ndim == 1, "y_dim must be a 1D array."
    expected_shape = (len(y_dim), len(x_dim))
    error_msg = (
        f"Data shape does not match dimensions. "
        f"Expected {expected_shape}, got data.shape={data.shape}."
    )
    assert data.shape == expected_shape, error_msg

    def gaussian_pdf_2d(
        coords_flat: np.ndarray,
        amplitude: float,
        mu_x: float,
        mu_y: float,
        sigma_x: float,
        sigma_y: float,
        rho: float,
    ) -> np.ndarray:
        """Helper function to calculate the 2D Gaussian PDF for a set of parameters."""
        rv = _params_to_multivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho)
        coords = coords_flat.reshape(-1, 2)

        return amplitude * rv.pdf(coords)

    # Setup the grid coordinates
    xx, yy = np.meshgrid(x_dim, y_dim)
    coords_flat = np.column_stack((xx.ravel(), yy.ravel()))
    z = data.ravel()

    # Setup a LMFit model object for the Gaussian PDF
    model = Model(gaussian_pdf_2d, independent_vars=["coords_flat"])

    params = Parameters()
    params.add("amplitude", value=np.max(z), min=0)
    params.add("mu_x", value=np.mean(x_dim))  # Initial guess for x mean
    params.add("mu_y", value=np.mean(y_dim))  # Initial guess for y mean
    params.add("sigma_x", value=1.0, min=0)
    params.add("sigma_y", value=1.0, min=0)
    params.add("rho", value=0.0, min=-1, max=1)  # Correlation coefficient

    # Fit the model to the data
    result = model.fit(z, params, coords_flat=coords_flat)

    return result


def find_peaks_from_pvalue(
    mip: torch.Tensor,
    scaled_mip: torch.Tensor,
    p_value_cutoff: float = 0.01,
    mask_radius: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finds the peak locations based on the p-value metric using mip and scaled mip.

    See the following for a reference on the p-value metric:
        https://journals.iucr.org/m/issues/2025/02/00/eh5020/eh5020.pdf

    Parameters
    ----------
    mip : torch.Tensor
        The maximum intensity projection (MIP) tensor from match template program.
    scaled_mip : torch.Tensor
        The z-score scaled MIP tensor from match template program.
    p_value_cutoff : float, optional
        The p-value cutoff for peak detection. Default is 0.01.
    mask_radius : float, optional
        The radius of the mask used to filter peaks. Default is 5.0.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The y and x coordinates of the detected peaks.
    """
    device = mip.device

    # Convert mip and scaled_mip tensors into numpy arrays
    scaled_mip = scaled_mip.cpu().numpy()
    mip = mip.cpu().numpy()

    # Apply the probit transformation to the quantiles of each of the data
    probit_zscore = probit_transform(scaled_mip.flatten())
    probit_mip = probit_transform(mip.flatten())

    # Dimensions for the data inferred from min/max values
    # NOTE: fixing the number of points used for the histogram here, could expose
    # options for fitting to the user...
    x_dim = np.linspace(
        start=probit_zscore.min(),
        stop=probit_zscore.max(),
        num=100,
    )
    y_dim = np.linspace(
        start=probit_mip.min(),
        stop=probit_mip.max(),
        num=100,
    )

    hist, _, _ = np.histogram2d(
        probit_zscore,
        probit_mip,
        # bins=(x_dim, y_dim),
        bins=(100, 100),
    )
    hist = np.ma.masked_array(hist, mask=hist == 0)

    # Fit the full covariance 2D Gaussian to the log transformed histogram
    # hist = np.masked_array(hist, mask=hist == 0)
    result = fit_full_cov_gaussian_2d(
        data=hist,
        x_dim=x_dim,
        y_dim=y_dim,
    )
    rv = _params_to_multivariate_normal(
        mu_x=result.params["mu_x"].value,
        mu_y=result.params["mu_y"].value,
        sigma_x=result.params["sigma_x"].value,
        sigma_y=result.params["sigma_y"].value,
        rho=result.params["rho"].value,
    )

    # Use numpy's masked array to only operate on points in the first quadrant
    points = np.column_stack((probit_zscore, probit_mip))
    # mask = (points[:, 0] < 0) | (points[:, 1] < 0)
    # points = np.ma.masked_array(points, mask=np.array([mask, mask]))

    # NOTE: This is a relatively slow step (~20-80s) since the cdf needs calculated for
    # large numbers of points. There are potential speedups like pre-filtering points
    # based on a less expensive bounds estimate, but that is left for future work.
    p_values = 1.0 - rv.cdf(points)

    p_values = p_values.reshape(mip.shape)

    # Convert back to Torch tensor and use the peak filtering utility function
    p_values = torch.from_numpy(p_values).to(device)
    peaks = torch.nonzero(p_values < p_value_cutoff, as_tuple=False)
    p_values = p_values[tuple(peaks.t())]

    if peaks.shape[0] == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    peaks = filter_peaks_by_distance(
        peak_values=p_values,
        peak_locations=peaks,
        distance_threshold=mask_radius,
    )

    return peaks[:, 0], peaks[:, 1]


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=duplicate-code
def extract_peaks_and_statistics_p_value(
    mip: torch.Tensor,
    scaled_mip: torch.Tensor,
    best_psi: torch.Tensor,
    best_theta: torch.Tensor,
    best_phi: torch.Tensor,
    best_defocus: torch.Tensor,
    correlation_average: torch.Tensor,
    correlation_variance: torch.Tensor,
    total_correlation_positions: int,
    p_value_cutoff: float = 0.01,
    mask_radius: float = 5.0,
) -> MatchTemplatePeaks:
    """Returns peak locations, stats, etc. using the pvalue metric.

    Parameters
    ----------
    mip : torch.Tensor
        Maximum intensity projection of the match template results.
    scaled_mip : torch.Tensor
        Scaled maximum intensity projection of the match template results.
    best_psi : torch.Tensor
        Best psi angles for each pixel.
    best_theta : torch.Tensor
        Best theta angles for each pixel.
    best_phi : torch.Tensor
        Best phi angles for each pixel.
    best_defocus : torch.Tensor
        Best relative defocus values for each pixel.
    correlation_average : torch.Tensor
        Average correlation value for each pixel.
    correlation_variance : torch.Tensor
        Variance of the correlation values for each pixel.
    total_correlation_positions : int
        Total number of correlation positions calculated during template matching. Must
        be provided if `z_score_cutoff` is not provided (needed for the noise model).
    p_value_cutoff : float, optional
        P-value cutoff value for peak detection. Default is 0.01.
    mask_radius : float, optional
        Radius for the mask used to filter peaks. Default is 5.0.

    Returns
    -------
    MatchTemplatePeaks
        A named tuple containing the peak locations, statistics, and other relevant
        data.
    """
    pos_y, pos_x = find_peaks_from_pvalue(
        mip=mip,
        scaled_mip=scaled_mip,
        p_value_cutoff=p_value_cutoff,
        mask_radius=mask_radius,
    )

    # Raise warning if no peaks are found
    if len(pos_y) == 0:
        warnings.warn("No peaks found using p-value metric.", stacklevel=2)

    # Extract peak heights, orientations, etc. from other maps
    return MatchTemplatePeaks(
        pos_y=pos_y,
        pos_x=pos_x,
        mip=mip[pos_y, pos_x],
        scaled_mip=scaled_mip[pos_y, pos_x],
        psi=best_psi[pos_y, pos_x],
        theta=best_theta[pos_y, pos_x],
        phi=best_phi[pos_y, pos_x],
        relative_defocus=best_defocus[pos_y, pos_x],
        correlation_mean=correlation_average[pos_y, pos_x],
        correlation_variance=correlation_variance[pos_y, pos_x],
        total_correlations=total_correlation_positions,
    )
