import streamlit as st
import requests
import tempfile
from PIL import Image
import tensorflow as tf   # ⬅️ Tambahkan baris ini
import torch
from torchvision import transforms
import cv2
import numpy as np
import tempfile, requests

# === Muat Model Klasifikasi ===
file_id_klasifikasi = "MASUKKAN_ID_FILE_H5"
url_klasifikasi = f"https://drive.google.com/uc?export=download&id={file_id_klasifikasi}"
r = requests.get(url_klasifikasi)
tmp_h5 = tempfile.NamedTemporaryFile(delete=False)
tmp_h5.write(r.content)
tmp_h5.flush()
klasifikasi_model = tf.keras.models.load_model(tmp_h5.name)

# === Muat Model Deteksi ===
file_id_deteksi = "MASUKKAN_ID_FILE_PT"
url_deteksi = f"https://drive.google.com/uc?export=download&id={file_id_deteksi}"
r = requests.get(url_deteksi)
tmp_pt = tempfile.NamedTemporaryFile(delete=False)
tmp_pt.write(r.content)
tmp_pt.flush()
deteksi_model = torch.load(tmp_pt.name, map_location=torch.device('cpu'))
deteksi_model.eval()


