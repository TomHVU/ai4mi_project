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

# Patient 27 has both good and bad
# TODO: import all patient images.
def import_image(source_path: Path) -> nibabel.Nifti1Image:
    # File location
    gt_path = source_path 
    assert gt_path.exists()

    # Import GT image
    gt_nifti = nibabel.load(gt_path)

    # # Increasing GT greyscale distance to increase contrast
    # gt_img *= 63
    # assert gt_img.dtype == np.uint8, gt_img.dtype
    # assert set(np.unique(gt_img)) <= set([0, 63, 126, 189, 252]), np.unique(gt_img)    

    return gt_nifti

def detamper(image, matrix, reverse=True) -> np.array:
    # Reverse the affine transformation using the inverse
    # Could be used as forward as well

    # If we want to reverse the transformation, use inverse
    if reverse:
        matrix = np.linalg.inv(matrix)

    # Perform transformation
    detampered_img = scipy.ndimage.affine_transform(image, matrix)

    # Return image
    return detampered_img

def main(src_path):
    step_by_step = args.step_by_step

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

    # To see separate matrix steps play out.
    if step_by_step:
        gt_heart_T4 = detamper(gt_heart, T4)
        gt_heart_T3 = detamper(gt_heart_T4, T3)
        gt_heart_R2 = detamper(gt_heart_T3, R2)
        gt_heart_T1 = detamper(gt_heart_R2, T1)

    # Reverse the affine transformation 
    gt_heart_combined = detamper(gt_heart, combined_affine)

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
    
    for patient_file in dir.iterdir():
        main(patient_file)