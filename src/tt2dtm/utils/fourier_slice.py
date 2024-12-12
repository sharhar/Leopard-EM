
import torch
from torch_fourier_slice.slice_extraction._extract_central_slices_rfft_3d import extract_central_slices_rfft_3d
from torch_grid_utils import fftfreq_grid


def multiply_by_sinc2(
    map: torch.Tensor,
) -> torch.Tensor:
    grid = fftfreq_grid(
        image_shape=map.shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=map.device
    )
    return map * torch.sinc(grid) ** 2

def fft_volume(
    volume: torch.Tensor,
    fftshift: bool = True,
) -> torch.Tensor:
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # volume center to array origin
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    if fftshift:
        dft = torch.fft.fftshift(dft, dim=(-3, -2,))  # actual fftshift of 3D rfft
    return dft
    
def extract_fourier_slice(
        dft_volume: torch.Tensor, 
        rotation_matrices: torch.Tensor, 
        volume_shape: tuple[int, int, int],
) -> torch.Tensor:

    # make projections by taking central slices
    projections = extract_central_slices_rfft_3d(
        volume_rfft=dft_volume,
        image_shape=volume_shape,
        rotation_matrices=rotation_matrices,
    )  # (..., h, w) rfft stack
    
    #FFT shift back to original because the ctf can only be applied to either rfft or fftshift
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of 2D rfft
    return projections