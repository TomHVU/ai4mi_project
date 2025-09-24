import random
from pathlib import Path
import argparse
from pathlib import Path
from multiprocessing import Pool
import nibabel
import numpy as np
import scipy.ndimage 
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm

def save_3d_image(image_3d, path) -> nibabel.Niftyimage:
    """
    Save the 3D slices as nifti image
    """
    pass
    
def main(args: argparse.Namespace):
    data_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)

    assert data_path.exists()
    assert dest_path.exists()

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