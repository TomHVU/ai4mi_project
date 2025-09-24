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
    
def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description = "Slicing parameters")

    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--dest_folder", type=int, default=42, help="Random seed")
    parser.add_argument("--num_classes", type=bool, default=False)
    parser.add_argument("--grp_regex")
    parser.add_argument("source_scan_pattern")

    args = parser.parse_args()
    random.seed(args.seed)

    return args

if __name__ == "__main__":
    args = get_args()
    dir = Path(args.source_dir)