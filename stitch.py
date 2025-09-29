import os
import sys
import re
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
from skimage.transform import resize
    
def main(args: argparse.Namespace):
    data_path: Path = Path(args.data_folder)
    dest_path: Path = Path(args.dest_folder)
    num_classes: int = int(args.num_classes)
    grp_regex: str = str(args.grp_regex)
    source_scan_pattern: str = str(args.source_scan_pattern)

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)

    assert data_path.exists()
    assert dest_path.exists()
    
    # The patient list
    # patients = [str(n).zfill(2) for n in range(1, 41)]

    patients = set()
    for parse in data_path.iterdir():
        match = re.search(grp_regex, parse.name)
        id = match.group(1)
        patients.add(id)

    # Sort numerically
    
    for id in patients:
        
        slices = []
        patient_files = [item for item in data_path.iterdir() if id in item.name]

        # Sort files numerically based on the number in filename
        def extract_number(file):
            parse = re.search(grp_regex, file.name)
            return int(parse.group(0)[-4:])  # convert to int for numerical sorting
        
        patient_files.sort(key=extract_number)

        # Load images
        for item in patient_files:
            number = extract_number(item)
            img = np.array(Image.open(f"{data_path}/{id}_{number:04d}.png")).astype(np.float32)
            # print(np.unique(img))
            img_scaled = np.round(img.astype(np.float32) * (num_classes - 1) / 255).astype(np.uint8)
            img_resized = resize(img_scaled, (512, 512), order=0, preserve_range=True, anti_aliasing=True)
            slices.append(img_resized)
            # slices.append(img_scaled)

        # slices = []
        # for item in data_path.iterdir():

        #     if id in item.name:
        #         parse = re.search(grp_regex, item.name)
        #         number = parse.group(0)[-4:]

        #         # Append per channel
        #         slices.append(resize(np.array(Image.open(f"{data_path}/{id}_{number}.png")), (512, 512), order=0, preserve_range=True, anti_aliasing=True))

        # Convert to np.array 
        # gt_img = np.array(slices, dtype=np.int8)
        gt_img = np.stack(slices, axis=2).astype(np.uint8)
        print(gt_img.shape)
        print(np.unique(gt_img))
        print(id)

        img = nibabel.load(Path(__file__).parent / source_scan_pattern.format(id_ = id))
        header = img.header
        affine = img.affine

        print(affine)

        gt_nifti = nibabel.Nifti1Image(gt_img, affine=affine, header=header)
        
        if not Path(f"{dest_path}/{id}").exists():
            os.mkdir(f"{dest_path}/{id}")
        nibabel.save(gt_nifti, f"{dest_path}/{id}/GT.nii.gz")
        print(f"Saved {id}'s stitched GT at {dest_path}/{id}/GT.nii.gz")
        break

def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description = "Slicing parameters")

    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--dest_folder", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--grp_regex", type=str, default="(Patient_\d\d)_\d\d\d\d")
    parser.add_argument("--source_scan_pattern", type=str, default="data/segthor_train/train/{id_}/GT.nii.gz")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")

    args = parser.parse_args()
    random.seed(args.seed)

    return args

if __name__ == "__main__":
    main(get_args())