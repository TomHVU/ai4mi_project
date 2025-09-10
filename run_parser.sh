# conda activate ai4mi

mkdir -p data/SEGTHOR_tmp/train/img
mkdir data/SEGTHOR_tmp/train/gt
mkdir -p data/SEGTHOR_tmp/val/img
mkdir data/SEGTHOR_tmp/val/gt

python slice_segthor.py \
--source_dir "data/segthor_train" \
--dest_dir "data/SEGTHOR_tmp" \
--retains 10 \
--seed 42
