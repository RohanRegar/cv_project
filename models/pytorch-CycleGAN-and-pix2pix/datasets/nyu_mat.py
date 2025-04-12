import h5py
import numpy as np
import cv2
import os
from tqdm import tqdm

# File path
file_path = '/DATA/soham/garments/preet/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/data/nyu_depth_v2_labeled.mat'

# Output directories
out_dir = '/DATA/soham/garments/preet/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/data/nyu_pngs_from_mat'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(out_dir, 'depth'), exist_ok=True)
os.makedirs(os.path.join(out_dir, 'label'), exist_ok=True)

with h5py.File(file_path, 'r') as f:
    rgb_data = f['images']
    depth_data = f['depths']
    label_data = f['labels']
    
    for i in tqdm(range(len(rgb_data))):
        # RGB image
        img = np.array(rgb_data[i]).transpose(1, 2, 0).astype(np.uint8)
        rgb_out = os.path.join(out_dir, 'rgb', f'{i:04d}.png')
        cv2.imwrite(rgb_out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert to BGR for OpenCV

        # Depth image (normalize to 0-255 for visualization)
        depth = np.array(depth_data[i])
        d_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_out = os.path.join(out_dir, 'depth', f'{i:04d}.png')
        cv2.imwrite(depth_out, d_norm.astype(np.uint8))

        # Label image
        label = np.array(label_data[i])
        label_out = os.path.join(out_dir, 'label', f'{i:04d}.png')
        cv2.imwrite(label_out, label.astype(np.uint8))
