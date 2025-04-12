import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests

st.write("Tumor Detection in Prostrate 👁️") 

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
else:
    image = Image.open("images/default_image.jpg")  # Use a default image from the 'images' folder

edges = cv2.Canny(np.array(image), 100, 200)
tab1, tab2 = st.tabs(["Detected edges", "Original"])
tab1.image(edges, use_column_width=True)
tab2.image(image, use_column_width=True)