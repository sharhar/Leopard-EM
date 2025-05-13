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
        #data = np.load(f"temp_ref_{i}_{j}.npy")
        
        data = np.load(f"test_data/temp_{i}_0_1.npy")
        data2 = np.load(f"test_data/temp_{i}_{j}_1.npy")

        diff_arr = np.abs(data - data2)

        np.save(f"test_data/diff_{i}_{j}_1.npy", diff_arr)
        
        #print(f"Sum {i} {j}:", np.sum(np.abs(data - data2)))
        #print(f"Mean {i} {j}:", np.mean(np.abs(data - data2)))
        print(f"Mean {i} {j}:", np.sum(data), np.sum(data2))

