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
    for j in range(7):
        data = np.load(f"corr_{i}_0.npy")
        data2 = np.load(f"corr_{i}_{j}.npy")

        #print(f"slice_cpu {i} shape: {data.shape}")
        print(f"{i} {j}:", np.sum(data), np.sum(data2), np.sum(np.abs(data2 - data)))

