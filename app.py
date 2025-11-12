import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import torch
from torchvision import transforms

# ================================
# Load Model
# ================================

@st.cache_resource
def load_all_models():
    st.sidebar.write("Memuat model...")
    klasifikasi_model = None
    deteksi_model = None

    # --- Muat Model Klasifikasi (.h5) ---
    try:
        klasifikasi_model = tf.keras.models.load_model("model/klasifikasi.h5", compile=False)
        klasifikasi_model.compile()
        st.sidebar.success("Model klasifikasi berhasil dimuat")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model klasifikasi: {e}")

    # --- Muat Model Deteksi (.pt) ---
    try:
        deteksi_model = torch.load("model/deteksi.pt", map_location=torch.device('cpu'))
        deteksi_model.eval()
        st.sidebar.success("Model deteksi berhasil dimuat")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model deteksi: {e}")
        
    return klasifikasi_model, deteksi_model

klasifikasi_model, deteksi_model = load_all_models()

# ================================
# Fungsi Prediksi
# ================================

CLASS_LABELS = ["Pensil", "Buku", "Penghapus", "Penggaris", "Lainnya"] 

def predict_klasifikasi(image):
    if klasifikasi_model is None:
        return "Model klasifikasi tidak tersedia."
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = klasifikasi_model.predict(img_array, verbose=0)
    class_idx = np.argmax(pred)
    prob = np.max(pred)
    
    label = CLASS_LABELS[class_idx] if class_idx < len(CLASS_LABELS) else f"Kelas {class_idx}"
    return f"**Hasil klasifikasi:** {label} (Probabilitas: {prob*100:.2f}%)"


def predict_deteksi(image):
    if deteksi_model is None:
        return "Model deteksi tidak tersedia."

    st.warning("Perhatian: Fungsi deteksi ini perlu diimplementasikan secara visual (menggambar bounding box) sesuai model PyTorch/YOLO yang digunakan.")

    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    img_tensor = transform(img_rgb).unsqueeze(0)
    
    with torch.no_grad():
        outputs = deteksi_model(img_tensor)
        

    return f"Hasil deteksi (Output Mentah): {outputs}"
    

# ================================
# Judul Dashboard
# ================================
st.title("Dashboard Klasifikasi & Deteksi Objek Alat Tulis")

# Pilih jenis prediksi
option = st.selectbox(
    "Pilih jenis prediksi",
    ("Klasifikasi Gambar", "Deteksi Objek")
)

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])


# ================================
# Tampilan Hasil
# ================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB") # Pastikan konversi ke RGB
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    if option == "Klasifikasi Gambar":
        st.subheader("Hasil Klasifikasi")
        hasil = predict_klasifikasi(image)
        st.markdown(hasil)
    else:
        st.subheader("Hasil Deteksi Objek")
        hasil = predict_deteksi(image)
        st.write(hasil)
