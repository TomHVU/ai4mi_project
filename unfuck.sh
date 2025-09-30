#!/bin/bash
# Base folder
BASE_DIR="/home/scur0607/ai4mi_project/data/SEGTHOR_STITCHED/val/pred"

# Loop over all Patient_* subfolders
for patient_dir in "$BASE_DIR"/Patient_*; do
    if [ -d "$patient_dir" ]; then
        patient_name=$(basename "$patient_dir")        # e.g., Patient_01
        src_file="$patient_dir/GT.nii.gz"
        dest_file="$BASE_DIR/${patient_name}.nii.gz"

        if [ -f "$src_file" ]; then
            echo "Moving $src_file -> $dest_file"
            mv "$src_file" "$dest_file"
            rmdir "$patient_dir"  # remove the now-empty folder
        else
            echo "Warning: $src_file not found"
        fi
    fi
done