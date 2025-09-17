import pickle
import random
import os
import sys
from pathlib import Path
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable
import nibabel
from PIL import Image
import numpy as np
import scipy.ndimage 
import skimage
import time

from utils import map_, tqdm_

# Define matrices
# T1
T1 = np.array([
    [1, 0, 0, 275],
    [0, 1, 0, 200],
    [0, 0, 1,   0],
    [0, 0, 0,   1]
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
T3 = np.linalg.inv(T1)

# T4
T4 = np.array([
    [1, 0, 0, 50],
    [0, 1, 0, 40],
    [0, 0, 1, 15],
    [0, 0, 0, 1]
], dtype=float)

# Put matrices into one affine matrix
combined_affine = T1 @ R2 @ T3 @ T4 

inverse_affine = np.linalg.inv(combined_affine)

# Patient 27 has both good and bad
# TODO: import all patient images.
def import_image(source_path: Path) -> nibabel.Nifti1Image:
    # File location
    gt_path = source_path 
    assert gt_path.exists()

    # Import GT image
    gt_nifti = nibabel.load(gt_path)

    return gt_nifti

def detamper(image, inverse_affine) -> np.array:
    # Reverse the affine transformation using the inverse
    # Could be used as forward as well

    # Perform transformation
    detampered_img = scipy.ndimage.affine_transform(image, inverse_affine)

    # Return image
    return detampered_img

def main(src_path):

    # Extract nifti file and save affine and header
    gt_img = import_image(src_path / "GT.nii.gz")
    gt_header = gt_img.header
    gt_affine = gt_img.affine

    # Get fdata to np.array(int16)
    gt_img = gt_img.get_fdata().astype(np.int16)

    # Isolate the heart channel
    # The heart is channel 2
    gt_heart = np.where(gt_img == 2, gt_img, 0)

    # Remove the tampered heart
    gt_img[gt_img == 2] = 0

    # Reverse the affine transformation 
    gt_heart_combined = scipy.ndimage.affine_transform(gt_heart, inverse_affine, order=0)

    # Put back the heart
    gt_img[gt_heart_combined == 2] = 2

     # Save image  
    gt_img = nibabel.Nifti1Image(gt_heart_combined, gt_affine, header=gt_header)
    nibabel.save(gt_img, src_path / 'GT_fixed.nii.gz')
    
def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description = "Slicing parameters")

    parser.add_argument("--source_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--step_by_step", type=bool, default=False)

    args = parser.parse_args()
    random.seed(args.seed)

    return args

if __name__ == "__main__":

    args = get_args()
    dir = Path(args.source_dir)

    t01 = time.time()

    for patient_file in dir.iterdir():
        t0 = time.time()
        main(patient_file)
        t1 = time.time()
        print(f"Patient {patient_file} transformed in {t1 - t0}")
        
    t11 = time.time()
    print(f"Completed 40 in {t11 - t01}")

    