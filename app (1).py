import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from scipy import ndimage

st.set_page_config(page_title="Analisis Gradien Gambar", layout="wide")

st.title("📊 Analisis Gradien Gambar & Matriks Kovarians")
st.markdown("""
Aplikasi ini mengimplementasikan proses ekstraksi fitur tekstur dari gambar menggunakan:
1. **Konversi ke Luminance** (menggunakan koefisien penglihatan manusia)
2. **Perhitungan Gradien** (turunan arah x dan y)
3. **Pembentukan Matriks Gradien**
4. **Perhitungan Matriks Kovarians** dari gradien
""")

# Sidebar untuk parameter
st.sidebar.header("⚙️ Parameter Analisis")
sample_method = st.sidebar.selectbox(
    "Metode Sampling:",
    ["Random Sampling", "Grid Sampling", "Semua Piksel"]
)

if sample_method == "Random Sampling":
    num_samples = st.sidebar.slider("Jumlah Sampel (N):", 100, 10000, 1000, 100)
elif sample_method == "Grid Sampling":
    grid_size = st.sidebar.slider("Ukuran Grid:", 5, 50, 20)
else:
    num_samples = None

show_vectors = st.sidebar.checkbox("Tampilkan Vektor Gradien", value=True)
vector_density = st.sidebar.slider("Kepadatan Vektor:", 1, 50, 15)

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar Anda:", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Konversi ke luminance
    if len(img_array.shape) == 3:
        # Gambar berwarna
        R = img_array[:,:,0].astype(float)
        G = img_array[:,:,1].astype(float)
        B = img_array[:,:,2].astype(float)
        L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    else:
        # Gambar grayscale
        L = img_array.astype(float)
    
    # Normalisasi luminance
    L = L / 255.0
    
    # Hitung gradien menggunakan filter Sobel
    Gx = ndimage.sobel(L, axis=1)  # gradien horizontal
    Gy = ndimage.sobel(L, axis=0)  # gradien vertikal
    
    # Membuat tab untuk visualisasi
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Gambar Asli", "⚫ Luminance", "📈 Gradien", "📊 Analisis"])
    
    with tab1:
        st.subheader("Gambar Asli")
        st.image(image, use_column_width=True)
    
    with tab2:
        st.subheader("Peta Luminance")
        fig_lum, ax_lum = plt.subplots(1, 2, figsize=(12, 4))
        
        ax_lum[0].imshow(L, cmap='gray')
        ax_lum[0].set_title('Luminance (Greyscale)')
        ax_lum[0].axis('off')
        
        ax_lum[1].hist(L.flatten(), bins=50, color='blue', alpha=0.7)
        ax_lum[1].set_title('Histogram Luminance')
        ax_lum[1].set_xlabel('Intensitas')
        ax_lum[1].set_ylabel('Frekuensi')
        
        st.pyplot(fig_lum)
        
        # Informasi statistik
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rata-rata Luminance", f"{L.mean():.4f}")
        with col2:
            st.metric("Std Dev Luminance", f"{L.std():.4f}")
        with col3:
            st.metric("Ukuran", f"{L.shape[0]} × {L.shape[1]}")
    
    with tab3:
        st.subheader("Analisis Gradien")
        
        # Hitung magnitudo dan arah gradien
        magnitude = np.sqrt(Gx**2 + Gy**2)
        direction = np.arctan2(Gy, Gx)
        
        fig_grad, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot Gx
        im1 = axes[0, 0].imshow(Gx, cmap='coolwarm')
        axes[0, 0].set_title('Gradien Horizontal (Gx)')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot Gy
        im2 = axes[0, 1].imshow(Gy, cmap='coolwarm')
        axes[0, 1].set_title('Gradien Vertikal (Gy)')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot magnitudo
        im3 = axes[0, 2].imshow(magnitude, cmap='hot')
        axes[0, 2].set_title('Magnitudo Gradien')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Plot arah
        im4 = axes[1, 0].imshow(direction, cmap='hsv')
        axes[1, 0].set_title('Arah Gradien (radian)')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Histogram magnitudo
        axes[1, 1].hist(magnitude.flatten(), bins=50, color='red', alpha=0.7)
        axes[1, 1].set_title('Histogram Magnitudo Gradien')
        axes[1, 1].set_xlabel('Magnitudo')
        axes[1, 1].set_ylabel('Frekuensi')
        
        # Scatter plot Gx vs Gy (subsampled)
        if show_vectors:
            h, w = Gx.shape
            stride = vector_density
            sample_x = Gx[::stride, ::stride].flatten()
            sample_y = Gy[::stride, ::stride].flatten()
            
            # Batasi jumlah titik untuk performa
            if len(sample_x) > 10000:
                idx = np.random.choice(len(sample_x), 10000, replace=False)
                sample_x = sample_x[idx]
                sample_y = sample_y[idx]
            
            axes[1, 2].scatter(sample_x, sample_y, alpha=0.3, s=1)
            axes[1, 2].set_title('Distribusi Gradien (Gx vs Gy)')
            axes[1, 2].set_xlabel('Gx')
            axes[1, 2].set_ylabel('Gy')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axhline(y=0, color='k', linestyle='-', alpha=0.2)
            axes[1, 2].axvline(x=0, color='k', linestyle='-', alpha=0.2)
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig_grad)
    
    with tab4:
        st.subheader("Matriks Kovarians Gradien")
        
        # Sampel titik-titik untuk matriks M
        height, width = Gx.shape
        
        if sample_method == "Random Sampling":
            # Random sampling
            indices = np.random.choice(height * width, min(num_samples, height * width), replace=False)
            Gx_flat = Gx.flatten()[indices]
            Gy_flat = Gy.flatten()[indices]
            
        elif sample_method == "Grid Sampling":
            # Grid sampling
            y_coords = np.arange(0, height, grid_size)
            x_coords = np.arange(0, width, grid_size)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
            Gx_flat = Gx[yy.flatten(), xx.flatten()]
            Gy_flat = Gy[yy.flatten(), xx.flatten()]
            
        else:  # Semua piksel
            Gx_flat = Gx.flatten()
            Gy_flat = Gy.flatten()
        
        # Batasi jumlah sampel untuk performa
        if len(Gx_flat) > 50000:
            indices = np.random.choice(len(Gx_flat), 50000, replace=False)
            Gx_flat = Gx_flat[indices]
            Gy_flat = Gy_flat[indices]
        
        N = len(Gx_flat)
        
        # Bentuk matriks M (N x 2)
        M = np.column_stack((Gx_flat, Gy_flat))
        
        # Hitung matriks kovarians C
        C = (1/N) * M.T @ M
        
        # Tampilkan hasil
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Matriks M (Sampel Gradien)")
            st.write(f"Dimensi: {M.shape} (N = {N}, 2 fitur)")
            
            # Tampilkan sebagian data
            if N > 10:
                st.dataframe(M[:10], use_container_width=True)
                st.caption(f"Menampilkan 10 dari {N} baris")
            else:
                st.dataframe(M, use_container_width=True)
        
        with col2:
            st.markdown("### Matriks Kovarians C")
            st.latex(r"C = \frac{1}{N} M^\top M")
            
            # Tampilkan matriks dengan format
            st.write("**Matriks Kovarians 2×2:**")
            st.write(f"$$C = \\begin{{bmatrix}} {C[0,0]:.6f} & {C[0,1]:.6f} \\\\ {C[1,0]:.6f} & {C[1,1]:.6f} \\end{{bmatrix}}$$")
            
            # Interpretasi
            st.markdown("#### Interpretasi:")
            st.write(f"1. **Varians Gx**: {C[0,0]:.6f} (penyebaran gradien horizontal)")
            st.write(f"2. **Varians Gy**: {C[1,1]:.6f} (penyebaran gradien vertikal)")
            st.write(f"3. **Kovarians**: {C[0,1]:.6f} (korelasi antara Gx dan Gy)")
            
            # Analisis tambahan
            trace = np.trace(C)
            determinant = np.linalg.det(C)
            st.write(f"4. **Trace (total varians)**: {trace:.6f}")
            st.write(f"5. **Determinant**: {determinant:.6f}")
        
        # Visualisasi ellipsoid kovarians
        st.markdown("### Visualisasi Elips Kovarians")
        
        fig_cov, ax = plt.subplots(figsize=(8, 8))
        
        # Plot sampel gradien
        if N > 10000:
            # Subsampling untuk visualisasi yang lebih jelas
            idx = np.random.choice(N, 10000, replace=False)
            ax.scatter(M[idx, 0], M[idx, 1], alpha=0.1, s=1, label='Sampel Gradien')
        else:
            ax.scatter(M[:, 0], M[:, 1], alpha=0.1, s=1, label='Sampel Gradien')
        
        # Plot elips kovarians
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        
        # Eigenvalues dan eigenvectors untuk elips
        eigvals, eigvecs = np.linalg.eigh(C)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        
        # Buat elips
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigvals)
        ellipse = Ellipse(xy=(0, 0), width=width, height=height, 
                         angle=angle, alpha=0.3, color='red', label='Elips Kovarians (1σ)')
        ax.add_patch(ellipse)
        
        ax.set_xlabel('Gx (Gradien Horizontal)')
        ax.set_ylabel('Gy (Gradien Vertikal)')
        ax.set_title('Distribusi Gradien dengan Elips Kovarians')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        st.pyplot(fig_cov)
        
        # Analisis tekstur berdasarkan matriks kovarians
        st.markdown("### Analisis Tekstur Gambar")
        
        ratio = C[0,0] / (C[1,1] + 1e-10)
        
        if determinant < 1e-6:
            texture_type = "**Tekstur Homogen/Flat** (gradien kecil di semua arah)"
        elif ratio > 2:
            texture_type = "**Dominan Gradien Horizontal** (banyak tepi vertikal)"
        elif ratio < 0.5:
            texture_type = "**Dominan Gradien Vertikal** (banyak tepi horizontal)"
        elif abs(C[0,1]) > 0.5 * np.sqrt(C[0,0] * C[1,1]):
            texture_type = "**Tekstur Terkorelasi** (gradien memiliki pola arah)"
        else:
            texture_type = "**Tekstur Kompleks** (gradien tersebar merata)"
        
        st.info(f"Berdasarkan matriks kovarians: {texture_type}")
        
        # Metrik tambahan
        st.markdown("#### Metrik Analisis:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            anisotropy = 1 - (min(eigvals) / (max(eigvals) + 1e-10))
            st.metric("Anisotropi", f"{anisotropy:.3f}")
        
        with col2:
            entropy_grad = -np.sum(eigvals * np.log(eigvals + 1e-10))
            st.metric("Entropi Gradien", f"{entropy_grad:.3f}")
        
        with col3:
            mean_magnitude = np.mean(np.sqrt(Gx_flat**2 + Gy_flat**2))
            st.metric("Magnitudo Rata-rata", f"{mean_magnitude:.3f}")

else:
    st.info("Silakan upload gambar untuk memulai analisis.")
    st.markdown("""
    ### Contoh Analisis yang Akan Dilakukan:
    
    1. **Gambar dengan banyak tepi vertikal** (misalnya pagar):
       - Gradien horizontal (Gx) akan besar
       - Matriks kovarians memiliki varians Gx yang dominan
    
    2. **Gambar dengan banyak tepi horizontal**:
       - Gradien vertikal (Gy) akan besar
       - Matriks kovarians memiliki varians Gy yang dominan
    
    3. **Gambar tekstur kompleks** (misalnya rumput):
       - Gradien tersebar merata
       - Kovarians antara Gx dan Gy mungkin tinggi
    """)

# Footer
st.markdown("---")
st.markdown("""
**Keterangan:** 
Aplikasi ini mengimplementasikan analisis struktur gambar berdasarkan:
1. **Luminance**: $$L(x, y) = 0.2126 R(x, y) + 0.7152 G(x, y) + 0.0722 B(x, y)$$
2. **Gradients**: $$G_x(x, y) = -\frac{\partial L}{\partial x}, \quad G_y(x, y) = -\frac{\partial L}{\partial y}$$
3. **Matriks Kovarians**: $$\mathbf{C} = \frac{1}{N} \mathbf{M}^\top \mathbf{M}$$
""")