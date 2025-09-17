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
#TODO: really?
T3 = np.linalg.inv(T1)

# T4
T4 = np.array([
    [1, 0, 0, 50],
    [0, 1, 0, 40],
    [0, 0, 1, 15],
    [0, 0, 0, 1]
], dtype=float)

# Patient 27 has both good and bad
# TODO: import all patient images.
def import_image(dest_path: Path, source_path: Path) -> nibabel.Nifti1Image:
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

def detamper(image, matrix, reverse=True):
    # Reverse the affine transformation

    # If we want to reverse the transformation, use inverse
    if reverse:
        matrix = np.linalg.inv(matrix)

    # Perform transformation
    detampered_img = scipy.ndimage.affine_transform(image, matrix)

    # Return image
    return detampered_img

def main(args: argparse.Namespace):
    #TODO: use single path, we put the file back anyways
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    gt_img = import_image(dest_path, src_path / "GT.nii.gz")

    gt_header = gt_nifti.header
    print(gt_header)
    gt_img = gt_nifti.get_fdata().astype(np.int16)
    affine = gt_nifti.affine
    
    # Isolate the heart channel
    # The heart is channel 2
    gt_heart = np.where(gt_img == 2, gt_img, 0)

    # Remove the incorrect channel
    #TODO: convert to np image
    gt_img[gt_img == 2] = 0

    # Perform reverse transformation
    # TODO: optimize performance

    combined_affine = T1 @ R2 @ T3 @ T4 

    print("Performing Reverse transformations")
    gt_heart_T4 = detamper(gt_heart, T4)
    gt_heart_T3 = detamper(gt_heart_T4, T3)
    gt_heart_R2 = detamper(gt_heart_T3, R2)
    gt_heart_T1 = detamper(gt_heart_R2, T1)

    gt_heart_combined = detamper(gt_heart, combined_affine)
    print("Transformations finished")

    # Put back the heart
    gt_img[gt_heart_T1 == 2] = 2

    # For testing purposes
    if patient_number == 27:

        # Import patient 27
        gt_img_27, patient_number, affine = import_image(dest_path, src_path / "GT2.nii.gz")

        print(skimage.metrics.normalized_mutual_information(gt_img, gt_img_27, bins=100))
        print(skimage.metrics.normalized_mutual_information(gt_img_27, gt_img_27, bins=100))

        # TODO: check if similar
        # assert np.array_equal(gt_img_27, gt_img)
       
    # Save image
    gt_img = nibabel.Nifti1Image(gt_img, affine)
    nibabel.save(gt_img, src_path / 'GT_detampered.nii.gz')

    # Save image
    gt_img = nibabel.Nifti1Image(gt_heart_T4, affine)
    nibabel.save(gt_img, src_path / 'GT_T4.nii.gz')

    # Save image
    gt_img = nibabel.Nifti1Image(gt_heart_T3, affine)
    nibabel.save(gt_img, src_path / 'GT_T3.nii.gz')
    
    # Save image  
    gt_img = nibabel.Nifti1Image(gt_heart_R2, affine)
    nibabel.save(gt_img, src_path / 'GT_R2.nii.gz')

     # Save image  
    gt_img = nibabel.Nifti1Image(gt_heart_combined, affine)
    nibabel.save(gt_img, src_path / 'GT_combined.nii.gz')
    
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Slicing parameters")

    parser.add_argument('--source_dir', type=str, default="banana")
    parser.add_argument('--dest_dir', type=str, default="banna")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    if args.dest_dir is None:
        args.dest_dir = args.source_dir

    return args

if __name__ == "__main__":
    main(get_args())