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
import scipy 

from utils import map_, tqdm_

# Patient 27 has both
def import_image(dest_path: Path, source_path: Path) -> tuple[float, float, float, float]:
    # File location
    gt_path = source_path / "GT.nii.gz"
    assert gt_path.exists()

    # Import GT image
    gt_img = nibabel.load(gt_path)

    # Get axii and determine heart channel
    # TODO: determine what channel is from the heart

    patient_number = 27
    return gt_img, patient_number

def detamper(image):
    # Reverse the affine transformation

    # T1
    T1 = np.array([
        [1, 0, 0,   0],
        [0, 1, 0,   0],
        [0, 0, 1,   0],
        [275, 200, 0, 1]
    ], dtype=float)

    # R2 (rotation about z-axis with φ = -(27/180)*π )
    phi = -(27/180) * np.pi
    R2 = np.array([
        [np.cos(phi), -np.sin(phi), 0, 0],
        [np.sin(phi),  np.cos(phi), 0, 0],
        [0,            0,           1, 0],
        [0,            0,           0, 1]
    ], dtype=float)

    # T3 = inverse of T1
    #TODO: really?
    T3 = np.linalg.inv(T1)

    # T4
    T4 = np.array([
        [1, 0, 0, 50],
        [0, 1, 0, 40],
        [0, 0, 1, 15],
        [0, 0, 0, 1]
    ], dtype=float)

    detampered_image = ""
    return detampered_image

def check_image(image, good_image):
    # Check if image has been transformed correctly

    assert mutual_information(image, good_image)

    return True

scipy.ndimage.affine_transform()

def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    gt_img, patient_number = import_image()
    print(gt_img.shape)

    # detampered_image = detamper(patient_image)

    if patient_number == 27:
        patient_27_good = import_image()
        check_image(gt_img, patient_27_good)

    print(src_path)
    print(dest_path)
    

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Slicing parameters")

    parser.add_argument('--source_dir', type=str, default="banana")
    parser.add_argument('--dest_dir', type=str, default="banna")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)
    print(args)

    if args.dest_dir is None:
        args.dest_dir = args.source_dir

    return args

if __name__ == "__main__":
    main(get_args())