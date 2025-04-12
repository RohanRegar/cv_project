import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_image(path):
    img = Image.open(path).convert('L')  # convert to grayscale
    return np.array(img).astype(np.float32) / 255.0  # normalize to [0, 1]

def evaluate_pair(real_img, fake_img):
    # Resize if needed to match dimensions
    if real_img.shape != fake_img.shape:
        raise ValueError(f"Shape mismatch: {real_img.shape} vs {fake_img.shape}")

    # Flatten for MAE and RMSE
    mae = mean_absolute_error(real_img.flatten(), fake_img.flatten())
    rmse = np.sqrt(mean_squared_error(real_img.flatten(), fake_img.flatten()))

    # SSIM with safe window size and grayscale setting
    h, w = real_img.shape
    min_dim = min(h, w)
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)  # odd and ≤ min_dim

    ssim_val = ssim(real_img, fake_img, data_range=1.0, win_size=win_size, channel_axis=None)
    return mae, rmse, ssim_val

def evaluate_folder(folder):
    mae_total, rmse_total, ssim_total = 0, 0, 0
    count = 0
    for file in os.listdir(folder):
        if file.endswith("_real_B.png"):
            base = file.replace("_real_B.png", "")
            real_path = os.path.join(folder, f"{base}_real_B.png")
            fake_path = os.path.join(folder, f"{base}_fake_B.png")
            if os.path.exists(fake_path):
                real = load_image(real_path)
                fake = load_image(fake_path)
                try:
                    mae, rmse, ssim_val = evaluate_pair(real, fake)
                    print(f"{base}: MAE={mae:.4f}, RMSE={rmse:.4f}, SSIM={ssim_val:.4f}")
                    mae_total += mae
                    rmse_total += rmse
                    ssim_total += ssim_val
                    count += 1
                except Exception as e:
                    print(f"Skipping {base} due to error: {e}")

    if count > 0:
        print("\n--- AVERAGE METRICS ---")
        print(f"MAE:  {mae_total / count:.4f}")
        print(f"RMSE: {rmse_total / count:.4f}")
        print(f"SSIM: {ssim_total / count:.4f}")
    else:
        print("No valid image pairs found.")

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--folder', required=True, help='Path to folder containing *_real_B.png and *_fake_B.png files')
    # args = parser.parse_args()
    folder='/DATA/soham/garments/preet/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/data/nyu_pngs_from_mat/results/images'
    evaluate_folder(folder)
