import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import gdown  # Ensure this is added to requirements.txt

# Define constants
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
NUM_CLASSES = 4
MODEL_PATH = 'models/512_X_512_True_att_unet_model_val_loss_0.4386.pth'

# Mapping class indices to colors
color_to_class = {
    (0, 0, 0): {"id": 0, "name": "Background (Black)"},
    (0, 0, 255): {"id": 1, "name": "Stroma (Blue)"},
    (0, 255, 0): {"id": 2, "name": "Benign (Green)"},
    (255, 255, 0): {"id": 3, "name": "Tumor (Yellow)"}
}
class_to_color = {v["id"]: k for k, v in color_to_class.items()}

# Define Attention Block
class AttentionBlock(torch.nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = torch.nn.Sequential(
            torch.nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )

        self.W_x = torch.nn.Sequential(
            torch.nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )

        self.psi = torch.nn.Sequential(
            torch.nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid()
        )

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Define Attention U-Net Model
class AttUNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(AttUNet, self).__init__()

        def CBR(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = CBR(1024, 512)

        self.upconv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = CBR(512, 256)

        self.upconv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = CBR(256, 128)

        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = CBR(128, 64)

        self.conv_last = torch.nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        enc4 = self.att4(g=dec4, x=enc4)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.att3(g=dec3, x=enc3)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.att2(g=dec2, x=enc2)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.att1(g=dec1, x=enc1)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        return self.conv_last(dec1)

# Function to download the model file from Hugging Face
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model file from Hugging Face...")
        url = "https://huggingface.co/RV23532/tumor-detection-in-prostrate-project-v0/resolve/main/512_X_512_True_att_unet_model_val_loss_0.4386.pth"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")

# Load the model
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        return None, None  # Indicate that the model is not yet available
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AttUNet(num_classes=NUM_CLASSES).to(device)
        print("Loading model state dictionary...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)
        model.eval()
        print("Model loaded successfully.")
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Perform inference
def perform_inference(model, device, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.to(device))
        output = F.softmax(output, dim=1)
        _, predicted_mask = torch.max(output, 1)
        return output, predicted_mask.squeeze(0).cpu().numpy()

# Map class indices to colors
def map_classes_to_colors(predicted_mask):
    height, width = predicted_mask.shape
    predicted_mask_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color_rgb in class_to_color.items():
        predicted_mask_image[predicted_mask == class_idx] = list(color_rgb)
    return predicted_mask_image

# Calculate class statistics
def calculate_class_statistics(output, predicted_mask):
    detected_classes = set(predicted_mask.flatten())
    confidences = output.squeeze(0).cpu().numpy()
    class_stats = []
    for class_idx in detected_classes:
        class_info = color_to_class[class_to_color[class_idx]]
        confidence = confidences[class_idx, predicted_mask == class_idx].mean()
        total_pixels = np.sum(predicted_mask == class_idx)
        class_stats.append({
            "Class Name": class_info["name"],
            "Confidence": f"{confidence:.2f}",
            "Pixels Detected": total_pixels
        })
    return class_stats