import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Deteksi Real vs AI", layout="wide")
st.title("🔬 Deteksi Gambar Real vs AI")
st.caption("Metode: Grayscale → Gradien → Struktur Kovarians")

# ===============================
# PREPROCESSING YANG BENAR
# ===============================
def preprocess_image(img_rgb):
    # RGB → Grayscale OpenCV (VALID)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Pastikan uint8
    gray = gray.astype(np.uint8)
    return gray

# ===============================
# GRADIENT (SOBEL)
# ===============================
def compute_gradients(gray):
    Gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return Gx, Gy

# ===============================
# KLASIFIKASI STRUKTURAL (VALID)
# ===============================
def classify_image(Gx, Gy):
    # Gradient structure
    G = np.vstack((Gx.flatten(), Gy.flatten()))
    C = np.cov(G)

    # Eigenvalues
    eigvals = np.linalg.eigvals(C)
    eigvals = np.sort(np.real(eigvals))[::-1]
    lambda1, lambda2 = eigvals

    ratio = lambda1 / (lambda2 + 1e-8)

    # Gradient magnitude entropy
    grad_mag = np.sqrt(Gx**2 + Gy**2)
    hist, _ = np.histogram(grad_mag, bins=256, density=True)
    hist += 1e-10
    entropy = -np.sum(hist * np.log2(hist))

    # ===============================
    # FINAL RULE (KONSISTEN)
    # ===============================
    if ratio > 3.0 and entropy < 6.5:
        label = "FAKE (AI-Generated Image)"
    else:
        label = "REAL (Natural Image)"

    return label, ratio, entropy, C, grad_mag

# ===============================
# UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload gambar", ["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    gray = preprocess_image(img_rgb)
    Gx, Gy = compute_gradients(gray)

    label, ratio, entropy, C, grad_mag = classify_image(Gx, Gy)

    # ===============================
    # DISPLAY
    # ===============================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Gambar Asli", use_column_width=True)

    with col2:
        st.image(gray, caption="Grayscale (Benar)", use_column_width=True)

    with col3:
        st.image(grad_mag, caption="Magnitude Gradien", clamp=True, use_column_width=True)

    st.markdown("---")
    st.subheader("📊 Analisis Statistik")

    col4, col5 = st.columns(2)
    with col4:
        st.metric("Eigenvalue Ratio (λ1 / λ2)", f"{ratio:.2f}")
    with col5:
        st.metric("Entropy Gradien", f"{entropy:.2f}")

    st.subheader("Matriks Kovarians")
    st.write(C)

    st.markdown("## 🧠 HASIL AKHIR")
    if "FAKE" in label:
        st.error(label)
    else:
        st.success(label)

    st.info(
        "Keputusan didasarkan pada struktur gradien global (anisotropi), "
        "bukan sekadar kekuatan tepi."
    )
