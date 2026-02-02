import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Deteksi Gambar Real vs AI",
    layout="wide"
)

st.title("🔍 Deteksi Gambar Real vs AI")
st.caption("Metode: Luminance → Gradien → Statistik (Std, Entropy, Kovarians)")

# ===============================
# FUNGSI LUMINANCE
# ===============================
def rgb_to_luminance(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return L.astype(np.float32)

# ===============================
# FUNGSI GRADIEN (SOBEL)
# ===============================
def compute_gradients(L):
    Gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    return Gx, Gy

# ===============================
# FUNGSI KLASIFIKASI (FIXED)
# ===============================
def classify_image(Gx, Gy):
    grad_mag = np.sqrt(Gx**2 + Gy**2)

    mean_grad = np.mean(grad_mag)
    std_grad = np.std(grad_mag)

    # Entropy gradien
    hist, _ = np.histogram(grad_mag, bins=256, density=True)
    hist += 1e-10
    entropy = -np.sum(hist * np.log2(hist))

    # Kovarians gradien
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    C = np.cov(np.vstack((Gx_flat, Gy_flat)))
    trace = np.trace(C)

    # ===============================
    # RULE BASED DECISION (VALID)
    # ===============================
    if std_grad < 12 and entropy < 4.5 and trace < 1500:
        label = "FAKE (Kemungkinan AI-Generated)"
    else:
        label = "REAL (Kemungkinan Foto Asli)"

    return label, mean_grad, std_grad, entropy, trace, C, grad_mag

# ===============================
# UPLOAD GAMBAR
# ===============================
uploaded_file = st.file_uploader(
    "Upload gambar (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Proses utama
    L = rgb_to_luminance(img_np)
    Gx, Gy = compute_gradients(L)

    result, mean_g, std_g, entropy, trace, C, grad_mag = classify_image(Gx, Gy)

    # ===============================
    # TAMPILAN GAMBAR
    # ===============================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Luminance (Grayscale)")
        st.image(L, clamp=True, use_column_width=True)

    with col3:
        st.subheader("Magnitude Gradien")
        st.image(grad_mag, clamp=True, use_column_width=True)

    # ===============================
    # HASIL ANALISIS
    # ===============================
    st.markdown("---")
    st.subheader("📊 Analisis Statistik")

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        st.metric("Mean Gradien", f"{mean_g:.2f}")
    with col5:
        st.metric("Std Gradien", f"{std_g:.2f}")
    with col6:
        st.metric("Entropy", f"{entropy:.2f}")
    with col7:
        st.metric("Trace Kovarians", f"{trace:.2f}")

    st.subheader("Matriks Kovarians")
    st.write(C)

    # ===============================
    # HASIL AKHIR
    # ===============================
    st.markdown("## 🧠 Prediksi Sistem")
    if "FAKE" in result:
        st.error(result)
    else:
        st.success(result)

    st.info(
        "Model ini menggunakan analisis statistik gradien sebagai baseline deteksi citra AI. "
        "Untuk akurasi lebih tinggi, disarankan integrasi Machine Learning."
    )

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Baseline AI Image Detection • Gradien & Kovarians • Streamlit")
