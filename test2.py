import numpy as np

def perimeter_sum(array):
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")
    top = array[0, :]
    bottom = array[-1, :]
    left = array[1:-1, 0]
    right = array[1:-1, -1]
    return np.sum(top) + np.sum(bottom) + np.sum(left) + np.sum(right) # - array[0, 0] - array[0, -1] - array[-1, 0] - array[-1, -1]

for i in range(4):
    #data = np.load(f"fourier_slice3_{i}.npy")
    data = np.load(f"cross_ref_{i}.npy")
    data2 = np.load(f"cross_test_{i}.npy")
    
    print(f"slice_cpu {i} shape: {data.shape}")
    print(np.sum(np.abs(data - data2)))
    print(np.mean(np.abs(data - data2)))
    #print(np.sum(data ** 2))
    #print(perimeter_sum(data) / (4 * data.shape[0] - 4))

