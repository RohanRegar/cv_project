# models/midas_depth.py
import cv2
import torch
import numpy as np
from PIL import Image
import io

def run(input_image=None):
    # Load model
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    midas.to('cpu')
    midas.eval()

    # Load transforms
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transforms.small_transform

    if input_image is None:
        # Webcam mode
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, None
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        # Uploaded image mode (PIL → numpy array)
        img_rgb = np.array(input_image.convert("RGB"))

    imgbatch = transform(img_rgb).to('cpu')

    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

    # Normalize and convert to uint8 image
    depth_min = output.min()
    depth_max = output.max()
    normalized = (255 * (output - depth_min) / (depth_max - depth_min)).astype(np.uint8)
    depth_img = Image.fromarray(normalized)
    input_img = Image.fromarray(img_rgb)

    return input_img, depth_img
