#!/usr/bin/env python3.7

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable
import nibabel
from PIL import Image
import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_


def norm_arr(img: np.ndarray) -> np.ndarray:
    # TODO: your code here

    raise NotImplementedError("Implement norm_arr")


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    assert set(np.unique(gt)) == set(range(5))

    return True


"""
TODO: Implement patient slicing.
Context:
  - Given an ID and paths, load the NIfTI CT volume and (if not test_mode) the GT volume.
  - Validate with sanity_ct / sanity_gt.
  - Normalise CT with norm_arr().
  - Slice the 3D volumes into 2D slices, resize to `shape`, and save PNGs.
  - Currently we have groundtruth masks marked as {0,1,2,3,4} but those values are hard to distinguish in a grayscale png.
    Multiplying by 63 maps them to {0,63,126,189,252}, which keeps labels visually distinct in a grayscale PNG.
    You can use the following code, which works for already sliced 2d images:
    gt_slice *= 63
    assert gt_slice.dtype == np.uint8, gt_slice.dtype
    assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)
  - Return the original voxel spacings (dx, dy, dz).

Hints:
  - Use nibabel to load NIfTI images.
  - Use skimage.transform.resize (tip: anti_aliasing might be useful)
  - The PNG files should be stored in the dest_path, organised into separate subfolders: train/img, train/gt, val/img, and val/gt
  - Use consistent filenames: e.g. f"{id_}_{idz:04d}.png" inside subfolders "img" and "gt"; where idz is the slice index.
"""

def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int], test_mode=False)\
        -> tuple[float, float, float]:

    id_path: Path = source_path / ("train" if not test_mode else "test") / id_
    ct_path: Path = (id_path / f"{id_}.nii.gz")
    assert id_path.exists()
    assert ct_path.exists()

    # --------- FILL FROM HERE -----------

    raise NotImplementedError("Implement slice_patient")


"""
TODO: Implement a simple train/val split.
Requirements:
  - List patient IDs from <src_path>/train (folder names).
  - Shuffle them (respect a seed set in main()).
  - Take the first `retains` as validation, and the rest as training.
  - Return (training_ids, validation_ids).
"""

def get_splits(src_path: Path, retains: int) -> tuple[list[str], list[str]]:
    # TODO: your code here

    raise NotImplementedError("Implement get_splits")

def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val", "test"], [training_ids, validation_ids, test_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 test_mode=mode == 'test')
        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                resolutions = list(map(pfun, iterator))
            case -1:
                resolutions = Pool().map(pfun, iterator)
            case _ as p:
                resolutions = Pool(p).map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
