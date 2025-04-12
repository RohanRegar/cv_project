
# import os
# import shutil
# from PIL import Image

# # Source folder containing the images and depth maps
# source_folder = '/DATA/soham/garments/preet/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/data/nyu2_train/basement_0001a_out'

# # Target folders for Option 2 format
# target_root = '/DATA/soham/garments/preet/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/data/'
# trainA_dir = os.path.join(target_root, 'trainA')
# trainB_dir = os.path.join(target_root, 'trainB')

# # Create output folders if they don't exist
# os.makedirs(trainA_dir, exist_ok=True)
# os.makedirs(trainB_dir, exist_ok=True)

# # Iterate through files in the source folder
# file_stem_set = set()
# for f in os.listdir(source_folder):
#     if f.endswith('.jpg'):
#         file_stem_set.add(os.path.splitext(f)[0])  # e.g., '1'
        
# # Process each pair
# for i, stem in enumerate(sorted(file_stem_set)):
#     rgb_path = os.path.join(source_folder, f'{stem}.jpg')
#     depth_path = os.path.join(source_folder, f'{stem}.png')
    
#     if os.path.exists(rgb_path) and os.path.exists(depth_path):
#         # You can resize or normalize if needed
#         rgb = Image.open(rgb_path).convert('RGB')
#         depth = Image.open(depth_path).convert('L')  # 1-channel grayscale
        
#         rgb.save(os.path.join(trainA_dir, f'{i:04d}.png'))
#         depth.save(os.path.join(trainB_dir, f'{i:04d}.png'))
#     else:
#         print(f"Missing pair for {stem}")

# print("✅ Conversion completed.")
import h5py

file_path = '/DATA/soham/garments/preet/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/data/nyu_depth_v2_labeled.mat'

with h5py.File(file_path, 'r') as f:
    # List all top-level keys (datasets/groups)
    print("Keys:", list(f.keys()))

    # Example: access a variable
    images = f['images']
    print("Shape of images:", images.shape)

