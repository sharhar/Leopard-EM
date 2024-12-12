
import torch
import torch.nn.functional as F

def is_efficient_size(n, radices):
    """Check if a number can be factored using only the given radices."""
    for radix in radices:
        while n % radix == 0:
            n //= radix
    return n == 1  # If n is reduced to 1, it's efficient

def next_efficient_size(n, radices):
    """Find the next number >= n that is efficient."""
    while not is_efficient_size(n, radices):
        n += 1
    return n

def calculate_optimal_padding(x, y, radices=(2, 3, 5, 7, 11, 13)):
    """
    Calculate the most efficient sizes to pad x and y to.
    
    Args:
        x (int): Current size of the first dimension.
        y (int): Current size of the second dimension.
        radices (tuple): Supported radices for FFT computation.

    Returns:
        tuple: Optimal sizes for x and y.
    """
    optimal_x = next_efficient_size(x, radices)
    optimal_y = next_efficient_size(y, radices)
    return optimal_x, optimal_y

def pad_volume(
    mrc_map: torch.Tensor, 
    pad_length: int,
):
    return F.pad(mrc_map, pad=[pad_length] * 6, mode='constant', value=0)