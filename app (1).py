import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Deteksi Real vs AI", layout="wide")
st.title("🔬 Deteksi Gambar Real vs AI (Gradient Structure Analysis)")

# ===============================
# LUMINANCE
# ===============================
def rgb_to_luminance(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32)

# ===============================
# GRADIENT
# ===============================
def compute_gradients(L):
    Gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    return Gx, Gy

# ===============================
# CLASSIFICATION (FINAL FIX)
# ===============================
def classify_image(Gx, Gy):
    # Flatten gradients
    G = np.vstack((Gx.flatten(), Gy.flatten()))

    # Covariance matrix
    C = np.cov(G)

    # Eigenvalue decomposition
    eigvals = np.linalg.eigvals(C)
    eigvals = np.sort(np.real(eigvals))[::-1]

    lambda1, lambda2 = eigvals

    # Eigenvalue ratio (anisotropy)
    ratio = lambda1 / (lambda2 + 1e-8)

    # Gradient magnitude stats (supporting, not main)
    grad_mag = np.sqrt(Gx**2 + Gy**2)
    entropy = -np.sum(
        np.histogram(grad_mag, bins=256, density=True)[0] * 
        np.log2(np.histogram(grad_mag, bins=256, density=True)[0] + 1e-10)
    )

    # ===============================
    # FINAL DECISION RULE
    # ===============================
    if ratio > 2.5 and entropy < 7.0:
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
    img = np.array(image)

    L = rgb_to_luminance(img)
    Gx, Gy = compute_gradients(L)

    label, ratio, entropy, C, grad_mag = classify_image(Gx, Gy)

    # ===============================
    # DISPLAY
    # ===============================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Gambar Asli", use_column_width=True)
    with col2:
        st.image(L, caption="Luminance", clamp=True, use_column_width=True)
    with col3:
        st.image(grad_mag, caption="Magnitude Gradien", clamp=True, use_column_width=True)

    st.markdown("---")
    st.subheader("📊 Statistik Struktural")

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
        "Keputusan utama didasarkan pada anisotropi gradien (rasio eigenvalue), "
        "yang terbukti kuat membedakan citra sintetis dan citra alami."
    )
