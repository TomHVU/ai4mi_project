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

from utils import map_, tqdm_

def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)
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