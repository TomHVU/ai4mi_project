conda activate ai4mi

# python stitch.py --data_folder /home/scur0607/ai4mi_project/data/SEGTHOR_CLEAN/train/gt --dest_folder /home/scur0607/ai4mi_project/data/SEGTHOR_STITCHED/train/gt 
# python stitch.py --data_folder /home/scur0607/ai4mi_project/data/SEGTHOR_CLEAN/val/gt --dest_folder /home/scur0607/ai4mi_project/data/SEGTHOR_STITCHED/val/gt 

python stitch.py --data_folder /home/scur0607/ai4mi_project/results/SEGTHOR/ce/iter023/val --dest_folder /home/scur0607/ai4mi_project/data/SEGTHOR_STITCHED/val/pred 

