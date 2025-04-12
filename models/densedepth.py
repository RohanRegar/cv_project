import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

class DenseDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.FPN(
            encoder_name='densenet121',
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    def trainable_encoder(self, trainable=True):
        for p in self.model.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

    def _num_params(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])

# ✅ Inference helper
model = DenseDepth()
model.load_state_dict(torch.load("models/densedepth_weights.pth", map_location=torch.device("cpu")))
model.eval()
model.to("cpu")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

def run(input_image=None):
    if input_image is None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, None
        input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    img_resized = transform(input_image).unsqueeze(0)  # 1x3x256x256
    with torch.no_grad():
        pred = model(img_resized.to("cpu"))
        pred = pred.squeeze().cpu().numpy()

    # Normalize for visualization
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    pred_vis = Image.fromarray(np.uint8(pred * 255))

    return input_image, pred_vis

# run("")

# import torch
# import torch.nn as nn
# import segmentation_models_pytorch as smp
# import torchvision.transforms as T
# from PIL import Image
# import numpy as np
# import cv2


# import gc
# gc.collect()

# class DenseDepth(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = smp.FPN(
#             encoder_name='densenet121',
#             encoder_weights="imagenet",
#             in_channels=3,
#             classes=1,
#         )

#     def forward(self, x):
#         return self.model(x)

# # 🔧 Load model and weights
# model = DenseDepth()
# model.load_state_dict(torch.load("models/densedepth_weights1.pth", map_location="cpu"))
# model.eval()
# model.to("cpu")

# # 🔧 Transform for input image
# transform = T.Compose([
#     T.Resize((256, 256)),
#     T.ToTensor(),
# ])

# # 🔍 Load image from file
# input_image = Image.open("im2.png").convert("RGB")  # <- replace with your test image
# input_tensor = transform(input_image).unsqueeze(0)  # shape: (1, 3, 256, 256)

# # 🔮 Run inference
# with torch.no_grad():
#     pred = model(input_tensor)
#     pred = pred.squeeze().cpu().numpy()

# # 🎨 Normalize depth map for visualization
# pred_norm = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
# depth_map = np.uint8(pred_norm * 255)

# # 💾 Save and show result
# depth_img = Image.fromarray(depth_map)
# depth_img.save("depth_output.png")
# depth_img.show()

# # Optionally show side-by-side using OpenCV
# cv2.imshow("Input", cv2.cvtColor(np.array(input_image.resize((256, 256))), cv2.COLOR_RGB2BGR))
# cv2.imshow("Depth", depth_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
