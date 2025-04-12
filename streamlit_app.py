import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import gdown  # To download from google drive

# Define constants
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
NUM_CLASSES = 4
MODEL_PATH = 'models/512_X_512_True_att_unet_model_val_loss_0.4386.pth'

# Mapping class indices to colors
color_to_class = {
    (0, 0, 0): 0,      # Background (Black)
    (0, 0, 255): 1,    # Stroma (Blue)
    (0, 255, 0): 2,    # Benign (Green)
    (255, 255, 0): 3   # Tumor (Yellow)
}
class_to_color = {v: k for k, v in color_to_class.items()}

# Define Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Define Attention U-Net Model
class AttUNet(nn.Module):
    def __init__(self, num_classes):
        super(AttUNet, self).__init__()
        self.num_classes = num_classes

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = CBR(3, 64)
        self.Conv2 = CBR(64, 128)
        self.Conv3 = CBR(128, 256)
        self.Conv4 = CBR(256, 512)
        self.Conv5 = CBR(512, 1024)

        self.Up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = CBR(1024, 512)

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = CBR(512, 256)

        self.Up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = CBR(256, 128)

        self.Up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = CBR(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

# Function to download the model file from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model file...")
        url = "https://drive.google.com/uc?id=1zGI0uvbOapmJ5UineFPt4yMKsuVrZyFu"
        gdown.download(url, MODEL_PATH, quiet=False)
        st.write("Model downloaded successfully.")

# Load the model
@st.cache_resource
def load_model():
    download_model()  # Ensure the model is downloaded before loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttUNet(num_classes=NUM_CLASSES).to(device)
    
    # Load the state dictionary with strict=False to handle mismatched keys
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    
    model.eval()
    return model, device

# Function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Function to perform inference
def perform_inference(model, device, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.to(device))
        output = F.softmax(output, dim=1)
        _, predicted_mask = torch.max(output, 1)
        return predicted_mask.squeeze(0).cpu().numpy()

# Function to map class indices to colors
def map_classes_to_colors(predicted_mask):
    height, width = predicted_mask.shape
    predicted_mask_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color_rgb in class_to_color.items():
        predicted_mask_image[predicted_mask == class_idx] = list(color_rgb)
    return predicted_mask_image

# Streamlit app
st.write("Tumor Detection in Prostrate 👁️")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
else:
    image = Image.open("images/default_image.png")  # Use a default image from the 'images' folder

# Display uploaded image
tab1, tab2, tab3, tab4 = st.tabs(["Uploaded Image", "Recalculated Original Image", "Predicted Mask", "Applied Mask on Image"])
tab1.image(image, use_column_width=True)

# Load model
model, device = load_model()

# Preprocess image and perform inference
image_tensor = preprocess_image(image)
predicted_mask = perform_inference(model, device, image_tensor)

# Map predicted mask to colors
predicted_mask_image = map_classes_to_colors(predicted_mask)

# Resize predicted mask to original image size
predicted_mask_image_pil = Image.fromarray(predicted_mask_image)
predicted_mask_image_pil = predicted_mask_image_pil.resize(image.size, resample=Image.Resampling.NEAREST)
predicted_mask_image_resized = np.array(predicted_mask_image_pil)

# Display predicted mask
tab3.image(predicted_mask_image_resized, use_column_width=True)

# Apply mask on the original image
applied_mask_image = np.array(image).copy()
alpha = 0.5  # Transparency factor
for class_idx, color_rgb in class_to_color.items():
    mask = np.all(predicted_mask_image_resized == np.array(list(color_rgb)), axis=-1)
    applied_mask_image[mask] = (alpha * np.array(list(color_rgb)) + (1 - alpha) * applied_mask_image[mask]).astype(np.uint8)

# Display applied mask
tab4.image(applied_mask_image, use_column_width=True)

