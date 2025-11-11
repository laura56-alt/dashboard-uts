# app.py
import streamlit as st
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
import cv2
import numpy as np

# ================================
# Load Model
# ================================
st.sidebar.title("Model")
st.sidebar.write("Memuat model...")

# Klasifikasi
klasifikasi_model = tf.keras.models.load_model("model/klasifikasi.h5")

# Deteksi
deteksi_model = torch.load("model/deteksi.pt", map_location=torch.device('cpu'))
deteksi_model.eval()

st.title("Dashboard Klasifikasi & Deteksi Objek Alat Tulis")

# Pilih jenis prediksi
option = st.selectbox(
    "Pilih jenis prediksi",
    ("Klasifikasi Gambar", "Deteksi Objek")
)

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

def predict_klasifikasi(image):
    img = image.resize((224,224))  # sesuaikan dengan input modelmu
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = klasifikasi_model.predict(img_array)
    class_idx = np.argmax(pred)
    return f"Hasil klasifikasi: Kelas {class_idx}"

def predict_deteksi(image):
    # Convert PIL ke array numpy
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Transform ke tensor
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)
    # Prediksi
    with torch.no_grad():
        outputs = deteksi_model(img_tensor)
    return f"Hasil deteksi: {outputs}"


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)
    
    if option == "Klasifikasi Gambar":
        hasil = predict_klasifikasi(image)
        st.write(hasil)
    else:
        hasil = predict_deteksi(image)
        st.write(hasil)
