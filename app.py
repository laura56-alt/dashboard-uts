import tempfile, requests

# URL langsung download Google Drive atau Hugging Face
url_klasifikasi = "LINK_DOWNLOAD_KLASIFIKASI"
response = requests.get(url_klasifikasi)
tmp_h5 = tempfile.NamedTemporaryFile(delete=False)
tmp_h5.write(response.content)
tmp_h5.flush()

klasifikasi_model = tf.keras.models.load_model(tmp_h5.name)

url_deteksi = "LINK_DOWNLOAD_DETEKSI"
response = requests.get(url_deteksi)
tmp_pt = tempfile.NamedTemporaryFile(delete=False)
tmp_pt.write(response.content)
tmp_pt.flush()

deteksi_model = torch.load(tmp_pt.name, map_location=torch.device('cpu'))

