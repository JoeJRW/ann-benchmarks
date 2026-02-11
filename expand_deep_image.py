"""
Expand deep-image-96-angular to 960 dimensions by repeating each 96-dim vector 10 times.

Since angular distance = 1 - dot(a,b)/(|a|*|b|), and tiling does not change it:
  dot(tile(a), tile(b)) = 10 * dot(a, b)
  |tile(a)| = sqrt(10) * |a|
  => angular_dist(tile(a), tile(b)) = angular_dist(a, b)

So neighbors and distances are identical to the original dataset.

Usage:
  1. Make sure data/deep-image-96-angular.hdf5 exists (download or run benchmark once)
  2. python expand_deep_image.py
"""

import os
import h5py
import numpy as np

from ann_benchmarks.datasets import get_dataset_fn, get_dataset

REPEAT = 10
SRC_DATASET = "deep-image-96-angular"
DST_DATASET = "deep-image-960-angular"

def main():
    # Ensure source dataset exists (downloads if needed)
    print(f"Loading source dataset: {SRC_DATASET}")
    src, _ = get_dataset(SRC_DATASET)

    train = np.array(src["train"])
    test = np.array(src["test"])
    neighbors = np.array(src["neighbors"])
    distances = np.array(src["distances"])
    distance_metric = src.attrs["distance"]

    print(f"Original train shape: {train.shape}")
    print(f"Original test shape:  {test.shape}")

    # Tile each vector 10 times: (N, 96) -> (N, 960)
    train_expanded = np.tile(train, REPEAT)
    test_expanded = np.tile(test, REPEAT)

    print(f"Expanded train shape: {train_expanded.shape}")
    print(f"Expanded test shape:  {test_expanded.shape}")

    # Write new dataset
    dst_fn = get_dataset_fn(DST_DATASET)
    print(f"Writing to {dst_fn}")

    with h5py.File(dst_fn, "w") as f:
        f.attrs["type"] = "dense"
        f.attrs["distance"] = distance_metric
        f.attrs["dimension"] = train_expanded.shape[1]
        f.attrs["point_type"] = src.attrs.get("point_type", "float")

        f.create_dataset("train", data=train_expanded)
        f.create_dataset("test", data=test_expanded)
        # Reuse ground truth directly (angular distance is unchanged after tiling)
        f.create_dataset("neighbors", data=neighbors)
        f.create_dataset("distances", data=distances)

    print(f"Done! Dataset saved to {dst_fn}")
    src.close()


if __name__ == "__main__":
    main()
