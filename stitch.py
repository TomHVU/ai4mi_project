import os
import re
import random
from pathlib import Path
import argparse
import nibabel
import numpy as np
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

    # Detect number of patients in dir using re.
    patients = set()
    for parse in data_path.iterdir():
        match = re.search(grp_regex, parse.name)
        id = match.group(1)
        patients.add(id)

    # Look through patient files.
    for id in patients:
        
        slices = []
        patient_files = [item for item in data_path.iterdir() if id in item.name]

        # Sort files numerically based on the number in filename.
        def extract_number(file):
            parse = re.search(grp_regex, file.name)
            return int(parse.group(0)[-4:])  # Convert to int for numerical sorting.
        
        # Sort by number
        patient_files.sort(key=extract_number)

        # Import GT mask, normalize and rescale.
        for item in patient_files:
            number = extract_number(item)
            img = np.array(Image.open(f"{data_path}/{id}_{number:04d}.png")).astype(np.float32)

            # Normalize.
            img_scaled = np.round(img.astype(np.float32) * (num_classes - 1) / 255).astype(np.uint8) # 63

            # Apply sklearn.transform.resize function.
            img_resized = resize(img_scaled, (512, 512), order=0, preserve_range=True, anti_aliasing=True)
            slices.append(img_resized)

        # Stack slices into 3D reconstruction.
        gt_img = np.stack(slices, axis=2).astype(np.uint8)

        # Find original CT image and grab affine and header.
        img = nibabel.load(Path(__file__).parent / source_scan_pattern.format(id_ = id))
        header = img.header
        affine = img.affine

        # Build nifty with 3D GT image, affine matrix and header.
        gt_nifti = nibabel.Nifti1Image(gt_img, affine=affine, header=header)
        
        # Save nifti.
        if not Path(f"{dest_path}/{id}").exists():
            os.mkdir(f"{dest_path}/{id}")
        nibabel.save(gt_nifti, f"{dest_path}/{id}/GT.nii.gz")
        print(f"Saved {id}'s stitched GT at {dest_path}/{id}/GT.nii.gz")

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