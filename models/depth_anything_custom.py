import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
import numpy as np

device = torch.device("cpu")  # Always run on CPU

class DepthAnythingCustom(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Use a pretrained encoder backbone (DenseNet or EfficientNet)
        self.encoder = timm.create_model('efficientnet_b0', features_only=True, pretrained=pretrained)

        # Get feature dimensions
        dummy_input = torch.randn(1, 3, 224, 224)
        features = self.encoder(dummy_input)
        feature_dims = [f.shape[1] for f in features]

        # Decoder with skip connections (FPN-like architecture)
        self.decoder = nn.ModuleDict()
        prev_dim = feature_dims[-1]

        # Top-down path with lateral connections
        for i in range(len(feature_dims)-2, -1, -1):
            # Lateral connection
            self.decoder[f'lateral_{i}'] = nn.Conv2d(feature_dims[i], 256, kernel_size=1)

            # Top-down connection
            self.decoder[f'top_down_{i}'] = nn.Sequential(
                nn.Conv2d(prev_dim, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

            # Update previous dimension
            prev_dim = 256

        # Final depth prediction head
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def trainable_encoder(self, trainable=True):
        # Enable or disable training for encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        # Get features from encoder
        features = self.encoder(x)

        # Top-down pathway with lateral connections (like FPN)
        prev_feature = features[-1]
        outputs = []

        for i in range(len(features)-2, -1, -1):
            # Get lateral feature
            lateral = self.decoder[f'lateral_{i}'](features[i])

            # Upsample previous feature
            top_down = nn.functional.interpolate(
                prev_feature, size=lateral.shape[2:], mode='bilinear', align_corners=False
            )

            # Apply convolution
            top_down = self.decoder[f'top_down_{i}'](top_down)

            # Combine lateral and top-down
            prev_feature = lateral + top_down
            outputs.append(prev_feature)

        # Use the final feature map for depth prediction
        final_feature = outputs[-1]

        # Apply depth head
        depth_pred = self.depth_head(final_feature)

        # Upsample to input size if needed
        if depth_pred.shape[2:] != x.shape[2:]:
            depth_pred = nn.functional.interpolate(
                depth_pred, size=x.shape[2:], mode='bilinear', align_corners=False
            )

        return depth_pred

    def _num_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


# Load model
model = DepthAnythingCustom(pretrained=False)
checkpoint = torch.load("models/depth_anything_weights.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().to(device)

# Define transforms
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def run(input_image):
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    output = model(input_tensor)[0, 0].cpu().numpy()
    
    # Convert to a visually friendly depth map
    depth_min = output.min()
    depth_max = output.max()
    output_vis = ((output - depth_min) / (depth_max - depth_min + 1e-8) * 255).astype(np.uint8)
    output_image = Image.fromarray(output_vis)

    return input_image, output_image
