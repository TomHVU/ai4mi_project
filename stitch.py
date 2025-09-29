import os
import sys
import regex as re
import random
from pathlib import Path
import argparse
from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict
import nibabel
import numpy as np
import scipy.ndimage 
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm
from PIL import Image
import sklearn


def import_images(data_path, grp_regex) -> np.uint8:
    """
    Import png images and stack them
    """
    

    pass


def save_3d_image(nifty_3d, path) -> nibabel.Niftyimage:
    """
    Save the 3D slices as nifti image
    """
    pass
    
def main(args: argparse.Namespace):
    data_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)
    num_classes: int = int(args.num_classes)
    grp_regex: str = str(args.grp_regex)
    source_scan_pattern: Path = Path(args.source_scan_pattern)

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)

    assert data_path.exists()
    assert dest_path.exists()
    
    # The patient list
    # patients = [str(n).zfill(2) for n in range(1, 41)]

    patients = list
    for parse in data_path.iterdir():
        parse = re.search(grp_regex, parse)
        id = parse.group(1)
        patients.append(id)

    # Sort numerically
    slices = list
    for id in patients:
        for item in data_path.iterdir():
            if f"Patient_{id}" in item:
                parse = re.search(grp_regex, item)
                number = parse.group(2)
                slices.append(sklearn.transform.resize(Image.open(f"{data_path}/Patient_{id}_{number}.png"), (512, 512)))
                #TODO: divide the channels by (255 / num_channels - 1)
        
        img = nibabel.load(source_scan_pattern.format(id))
        header = img.header
        affine = img.affine

        gt_stitched = np.array(slices)

        gt_nifti = nibabel.Nifti1Image(gt_stitched, affine=affine, header=header)
        nibabel.save(gt_nifti, f"{dest_path}/stitched/GT.nii.gz")


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description = "Slicing parameters")

    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--dest_folder", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=255)
    parser.add_argument("--grp_regex", type=str, default="(Patient_\d\d)_\d\d\d\d")
    parser.add_argument("--source_scan_pattern", type=str, default="data/segthor_train/train/{id_}/GT.nii.gz")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")

    args = parser.parse_args()
    random.seed(args.seed)

    return args

if __name__ == "__main__":
    main(get_args)