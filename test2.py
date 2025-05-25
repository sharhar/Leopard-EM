import numpy as np

import mrcfile

def read_mrc(filename):
    with mrcfile.open(filename, permissive=True) as mrc:
        data = mrc.data
        return data
    
# ref_dat = read_mrc("data/data2/results_ref/output_scaled_mip.mrc")
# dat = read_mrc("data/data2/results/output_scaled_mip.mrc")

# np.save("diff2.npy", ref_dat - dat)

def perimeter_sum(array):
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")
    top = array[0, :]
    bottom = array[-1, :]
    left = array[1:-1, 0]
    right = array[1:-1, -1]
    return np.sum(top) + np.sum(bottom) + np.sum(left) + np.sum(right) # - array[0, 0] - array[0, -1] - array[-1, 0] - array[-1, -1]

for i in range(4):
    for j in range(13):
        data = np.load(f"corr_ref_{i}_{j}.npy")
        data2 = np.load(f"corr_{i}_{j}.npy")

        np.save(f"diff_{i}_{j}.npy", data-data2)

        #print(f"slice_cpu {i} shape: {data.shape}")
        print(f"{i} {j}:") #, np.sum(data), np.sum(data2), np.sum(np.abs(data-data2)))

        print(f"\tperimeter_sum ref: {perimeter_sum(data)}")
        print(f"\tperimeter_sum    : {perimeter_sum(data2)}")
        print(f"\tsum ref: {np.sum(np.abs(data))}")
        print(f"\tsum    : {np.sum(np.abs(data2))}")
        print(f"\tdiff sum: {np.sum(data-data2)}")
        print(f"\tdiff abs sum: {np.sum(np.abs(data-data2))}")

        max_value = np.max(np.abs(data-data2))

        print(f"\tdiff max: {max_value}")
        max_value_location = np.unravel_index(np.argmax(np.abs(data-data2)), data.shape)

        value_at_max = data[max_value_location]

        #print(f"\tmax value location: {max_value_location}")
        print(f"\tdiff max ref value: {value_at_max}")

        # print(f"\tdiff max ratio: {np.max(np.abs(data-data2) / np.abs(data))}")

        # max_ratio_location = np.unravel_index(np.argmax(np.abs(data-data2) / np.abs(data)), data.shape)
        # value_at_max_ratio = data[max_ratio_location]
        # value_2_at_max_ratio = data2[max_ratio_location]

        # print(f"\tmax ratio location: {max_ratio_location}")
        # print(f"\tmax ratio value: {value_at_max_ratio}")
        # print(f"\tmax ratio value2: {value_2_at_max_ratio}")

        print(f"\tmax ref: {np.max(np.abs(data))}")
        print(f"\tmax    : {np.max(np.abs(data2))}")

