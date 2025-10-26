import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# Judul Dashboard
st.title("Halo 👋 Ini Dashboard Pertamaku!")
st.write("Selamat datang di dashboard Streamlit 🚀")

# Slider untuk memilih panjang data
n = st.slider("Pilih jumlah data", min_value=3, max_value=10, value=5)

# Data contoh
data = [i * 5 for i in range(1, n + 1)]

# Buat grafik
fig, ax = plt.subplots()
ax.plot(data, marker='o', color='#00FF00', linewidth=2)  # hijau terang
ax.set_title(f"Grafik Contoh dengan {n} Titik Data (Matplotlib)")
ax.set_xlabel("Index")
ax.set_ylabel("Nilai")

# Simpan figure langsung ke memori (bukan file)
buf = BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)

# Tampilkan gambar langsung dari memori
st.image(buf, caption="Grafik Contoh (Hijau) 💚", use_container_width=True)

