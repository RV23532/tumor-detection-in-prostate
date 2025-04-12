import streamlit as st
from PIL import Image
import numpy as np
from backend import (
    load_model,
    preprocess_image,
    perform_inference,
    map_classes_to_colors,
    calculate_class_statistics,
    class_to_color,
    download_model  # Import download_model
)

# Streamlit app
st.title("Tumor Detection in Prostate 👁️")

# Check if the model is available
with st.spinner("Checking for model file..."):
    model, device = load_model()
    if model is None:
        st.warning("Model file not found. Downloading now...")
        with st.spinner("Downloading model file. Please wait..."):
            download_model()
        st.success("Model downloaded successfully!")
        model, device = load_model()
        if model is None:
            st.error("Failed to load the model. Please try again.")
            st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
else:
    st.warning("Please upload an image to proceed.")
    st.stop()

# Preprocess image and perform inference
image_tensor = preprocess_image(image)
output, predicted_mask = perform_inference(model, device, image_tensor)

# Map predicted mask to colors
predicted_mask_image = map_classes_to_colors(predicted_mask)

# Resize predicted mask to original image size
predicted_mask_image_pil = Image.fromarray(predicted_mask_image)
predicted_mask_image_pil = predicted_mask_image_pil.resize(image.size, resample=Image.Resampling.NEAREST)
predicted_mask_image_resized = np.array(predicted_mask_image_pil)

# Apply mask on the original image
applied_mask_image = np.array(image).copy()
alpha = 0.5  # Transparency factor
for class_idx, color_rgb in class_to_color.items():
    mask = np.all(predicted_mask_image_resized == np.array(list(color_rgb)), axis=-1)
    applied_mask_image[mask] = (alpha * np.array(list(color_rgb)) + (1 - alpha) * applied_mask_image[mask]).astype(np.uint8)

# Display tabs
tab1, tab2, tab3 = st.tabs(["Original Image", "Predicted Mask", "Applied Mask"])
tab1.image(image, caption="Original Image", use_container_width=True)
tab2.image(predicted_mask_image_resized, caption="Predicted Mask", use_container_width=True)
tab3.image(applied_mask_image, caption="Applied Mask", use_container_width=True)

# Calculate and display class statistics
class_stats = calculate_class_statistics(output, predicted_mask)
st.write("### Detected Classes, Confidence Scores, and Total Pixels")
st.table(class_stats)