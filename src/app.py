import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import io
import math
import time
import subprocess # <--- Added for launching Windows apps
from torchvision import transforms
from scipy.fftpack import dct
import matplotlib.cm as cm

# --- Imports for Metrics ---
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# --- Import AI Model ---

try:
    from AI_functions import SatelliteVBR, NUM_FILTERS_N, NUM_FILTERS_M
except ImportError:
    st.error("CRITICAL ERROR: 'AI_functions.py' not found.")
    st.stop()

# --- Constants & Paths ---
MODEL_PATH = "../models/final_model.pth"
PATCH_SIZE = 64 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CROSS-PLATFORM CONFIGURATION ---
# 1. Path to MATLAB executable (Windows version accessed via WSL mount)
MATLAB_EXE = "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe"

# 2. The Project Folder as seen by WSL (Where we write the .bin files)
WSL_PROJECT_PATH = "/mnt/d/Core_Folder-Om/minor_project/cross_os"

# 3. The Project Folder as seen by Windows (Where MATLAB starts)
WIN_PROJECT_PATH = r"D:\Core_Folder-Om\minor_project\cross_os"

# 4. The Script Name (Must exist in D:\Core_Folder-Om\minor_project)
MATLAB_SCRIPT_NAME = "SatelliteSlicingVis"

# --- Session State Initialization ---
if 'log_history' not in st.session_state:
    st.session_state['log_history'] = []
if 'step' not in st.session_state:
    st.session_state['step'] = 1

# --- Helper Functions ---

def update_log(message, type='info'):
    """Adds a timestamped message to the sidebar log."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state['log_history'].append({"time": timestamp, "msg": message, "type": type})

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = SatelliteVBR(N=NUM_FILTERS_N, M=NUM_FILTERS_M)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        update_log("AI Model Loaded Successfully", "success")
    except Exception as e:
        update_log(f"Model Load Failed: {e}", "error")
        return None
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def calculate_bpp(likelihoods, num_pixels):
    total_bits = 0
    for like in likelihoods.values():
        total_bits += torch.log(like).sum() / (-math.log(2))
    return total_bits.item() / num_pixels

def run_inference(model, x):
    with torch.no_grad():
        # Encoder
        y = model.g_a(x)
        z = model.h_a(y)
        
        # Latent/Reconstruction
        z_hat, z_likelihoods = model.entropy_bottleneck(z)
        mean_scale = model.h_s(z_hat)
        mean, log_scale = mean_scale.chunk(2, 1)
        scale = torch.exp(log_scale)
        y_hat, y_likelihoods = model.gaussian_conditional(y, scale, mean)
        x_hat = model.g_s(y_hat).clamp(0, 1)
        
        # BPP
        num_pixels = x.size(2) * x.size(3)
        bpp = calculate_bpp({"y": y_likelihoods, "z": z_likelihoods}, num_pixels)
        
    return x_hat, y, z, bpp

def compress_jpeg_match_bpp(image_pil, target_bpp):
    width, height = image_pil.size
    num_pixels = width * height
    best_img = image_pil
    min_diff = float('inf')
    
    # Binary search-ish approach for speed
    for q in range(1, 96, 2):
        buf = io.BytesIO()
        image_pil.save(buf, format="JPEG", quality=q)
        current_bpp = (buf.tell() * 8) / num_pixels
        diff = abs(current_bpp - target_bpp)
        if diff < min_diff:
            min_diff = diff
            best_img = Image.open(buf).convert("RGB")
            buf.seek(0)
    return best_img

def get_metrics(orig, comp):
    img1 = np.array(orig)
    img2 = np.array(comp)
    if SKIMAGE_AVAILABLE:
        p = psnr(img1, img2)
        s = ssim(img1, img2, channel_axis=2, win_size=3)
    else:
        # Simple Fallback
        mse = np.mean((img1 - img2)**2)
        p = 20 * math.log10(255.0 / math.sqrt(mse)) if mse != 0 else 100
        s = 0
    return p, s

def get_dct_spectrum(image_pil):
    """
    Computes the 2D DCT and returns a COLORFUL heatmap (Jet Colormap).
    Blue areas = Low Energy (Blurred/High Frequency removed).
    Red/Yellow areas = High Energy (Structural Information).
    """
    # 1. Convert to Grayscale
    img_gray = image_pil.convert('L')
    img_array = np.array(img_gray)

    # 2. Compute 2D DCT (Scipy)
    dct_rows = dct(img_array, type=2, axis=0, norm='ortho')
    dct_2d = dct(dct_rows, type=2, axis=1, norm='ortho')

    # 3. Log Transform (to visualize the massive range of values)
    log_dct = np.log1p(np.abs(dct_2d))

    # 4. Normalize to 0-1 Range
    min_val = log_dct.min()
    max_val = log_dct.max()
    
    # Avoid divide by zero
    if max_val - min_val == 0:
        norm_dct = np.zeros_like(log_dct)
    else:
        norm_dct = (log_dct - min_val) / (max_val - min_val)

    # 5. Apply "Jet" Colormap (Blue background for high freq loss)
    # This creates an RGBA array (Height, Width, 4)
    colormap = cm.get_cmap('jet') 
    colored_dct = colormap(norm_dct)

    # 6. Convert to PIL Image (Drop Alpha channel, scale to 0-255)
    img_out = Image.fromarray((colored_dct[:, :, :3] * 255).astype(np.uint8))

    return img_out

# --- UI Layout ---

st.set_page_config(page_title="DSP-Aware 5G Satellite AI", layout="wide")

# Sidebar: System Updates
with st.sidebar:
    st.header("🖥️ System Updates")
    st.divider()
    
    # Load Model Immediately
    model = load_model()
    
    # Render Log History
    if st.session_state['log_history']:
        for log in st.session_state['log_history']:
            if log['type'] == 'success':
                st.success(f"[{log['time']}] {log['msg']}")
            elif log['type'] == 'error':
                st.error(f"[{log['time']}] {log['msg']}")
            else:
                st.info(f"[{log['time']}] {log['msg']}")
    else:
        st.write("waiting for initialization...")

# Main Title & Description
st.title("DSP-Aware Convolutional Autoencoder for Satellite Image Compression in 5G Network Slicing")

st.markdown("""
> **System Description:**  
> - This AI-driven system employs a **hyperprior autoencoder** to generate highly compressed yet **high-fidelity satellite images**, preserving essential **high-frequency details** with performance comparable to advanced mathematical codecs and leverages **5G Network Slicing** to enable **efficient**, **mission-critical**, and **ultra-low-latency** data transmission.
""")

st.divider()


# --- STEP 1: Image Capture ---
st.header("1. 🛰️ Image Captured by Satellite")
uploaded_file = st.file_uploader("Select Satellite Imagery Source", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Handle Image Loading
    if 'current_file_name' not in st.session_state or st.session_state['current_file_name'] != uploaded_file.name:
        st.session_state['current_file_name'] = uploaded_file.name
        st.session_state['original_image'] = Image.open(uploaded_file).convert("RGB")
        update_log("Satellite Image Captured", "info")
        st.session_state['step'] = 2 # Advance step

    # Display Image (Reduced size as requested)
    st.image(st.session_state['original_image'], caption="Raw Satellite Input", width=200)

    # --- STEP 2: Compression ---
    st.divider()
    st.header("2. 📉 Generate Compression Data (Latent Vectors)")
    
    if st.button("⚡ Generate Image Vector & Key"):
        with st.spinner("Running DSP-Aware Autoencoder..."):
            # Run Model
            img_tensor = preprocess_image(st.session_state['original_image'])
            recon, y, z, bpp = run_inference(model, img_tensor)
            
            # Store Data
            st.session_state['recon_tensor'] = recon
            st.session_state['y_data'] = y.cpu().numpy()
            st.session_state['z_data'] = z.cpu().numpy()
            st.session_state['ai_bpp'] = bpp
            
            update_log("Latent Vector (y) Generated", "success")
            update_log("Probability Model Key (z) Generated", "success")
            st.session_state['step'] = 3

    # Show Download Options if Data Exists
    if st.session_state.get('step') >= 3:
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("💾 Download Latent (y.bin)", st.session_state['y_data'].tobytes(), "y.bin")
        with c2:
            st.download_button("💾 Download Key (z.bin)", st.session_state['z_data'].tobytes(), "z.bin")

    # --- STEP 3: 5G Transmission (UPDATED FOR WINDOWS MATLAB) ---
    if st.session_state.get('step') >= 3:
        st.divider()
        st.header("3. 📡 5G Network Slicing Transmission")
        
        col_trans, col_dummy = st.columns([1, 2])
        with col_trans:
            if st.button("🚀 Transmit via Network Slice"):
                
                # 1. Save files to the SHARED FOLDER (D:/...)
                # Ensure the directory exists (from WSL perspective)
                if not os.path.exists(WSL_PROJECT_PATH):
                    # Try to create it, though it should usually exist
                    try:
                        os.makedirs(WSL_PROJECT_PATH)
                    except PermissionError:
                        st.error(f"Permission Error: Cannot write to {WSL_PROJECT_PATH}")
                        st.stop()

                y_path = os.path.join(WSL_PROJECT_PATH, "y.bin")
                z_path = os.path.join(WSL_PROJECT_PATH, "z.bin")

                st.session_state['y_data'].tofile(y_path)
                st.session_state['z_data'].tofile(z_path)
                
                update_log(f"Data packets transmitted to {WSL_PROJECT_PATH}", "info")
                
                # 2. Launch Windows MATLAB via Subprocess
                try:
                    with st.spinner("Launching Windows MATLAB Simulation..."):
                        # Construct the command to run Windows MATLAB
                        # -sd sets the Startup Directory (so it finds your .m script and .bin files)
                        # -r runs the script immediately
                        cmd = [
                            MATLAB_EXE,
                            "-sd", WIN_PROJECT_PATH,
                            "-r", f"run('{MATLAB_SCRIPT_NAME}');",
                            "-nologo"
                        ]
                        
                        # Popen launches it as a separate process
                        subprocess.Popen(cmd)
                        
                        update_log("Transmitted via Network Slicing (MATLAB)", "success")
                        st.success("✅ Windows MATLAB Launched! Please check your taskbar.")
                        
                except FileNotFoundError:
                    st.error(f"Could not find MATLAB at {MATLAB_EXE}. Check the path.")
                except Exception as e:
                    st.error(f"Error launching MATLAB: {e}")
                
                st.session_state['step'] = 4

    # --- STEP 4: Ground Station ---
    if st.session_state.get('step') >= 4:
        st.divider()
        st.header("4. 🌎 Ground Station: Decode & Compare")
        
        if st.button("🔍 Run AI System & Compare Results"):
            with st.spinner("Decoding and calculating Metrics..."):
                # Prepare Images
                orig_pil = st.session_state['original_image'].resize((PATCH_SIZE, PATCH_SIZE))
                recon_pil = transforms.ToPILImage()(st.session_state['recon_tensor'].squeeze().cpu())
                jpeg_pil = compress_jpeg_match_bpp(orig_pil, st.session_state['ai_bpp'])
                
                # Calculate Metrics
                p_ai, s_ai = get_metrics(orig_pil, recon_pil)
                p_jpg, s_jpg = get_metrics(orig_pil, jpeg_pil)
                
                st.session_state['results'] = {
                    "orig": orig_pil, "recon": recon_pil, "jpeg": jpeg_pil,
                    "p_ai": p_ai, "s_ai": s_ai,
                    "p_jpg": p_jpg, "s_jpg": s_jpg
                }
                update_log("Image Decoded & Verified", "success")

        # Display Results
        if 'results' in st.session_state:
            res = st.session_state['results']
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.subheader("Original")
                st.image(res['orig'], width=250)
            
            with c2:
                st.subheader("AI Reconstructed")
                st.image(res['recon'], width=250)
                st.success(f"**PSNR:** {res['p_ai']:.2f} dB\n\n**SSIM:** {res['s_ai']:.4f}")
            
            with c3:
                st.subheader("JPEG (Same BPP)")
                st.image(res['jpeg'], width=250)
                st.warning(f"**PSNR:** {res['p_jpg']:.2f} dB\n\n**SSIM:** {res['s_jpg']:.4f}")
            
            st.info(f"**Compression Rate (BPP):** {st.session_state['ai_bpp']:.4f} bits/pixel")
            
            
            # --- STEP 5: DCT Analysis ---
            if st.session_state.get('results'):
                st.divider()
                st.header("5. 🧮 Image Processing - Discrete Cosine Transform")
                
                st.markdown("""
                **Frequency Domain Analysis:**
                * **Blue Zones:** Represent areas with little to no energy (High frequencies that have been blurred/discarded).
                * **Red/Yellow Zones:** Represent the core structural information of the image.
                """)

                if st.button("📊 Calculate Frequency Spectrum"):
                    with st.spinner("Computing Colorful 2D-DCT transforms..."):
                        # Retrieve images from previous step
                        res = st.session_state['results']
                        
                        # Compute Spectrums
                        dct_ai = get_dct_spectrum(res['recon'])
                        dct_jpeg = get_dct_spectrum(res['jpeg'])
                        
                        # Layout
                        c_dct1, c_dct2 = st.columns(2)
                        
                        with c_dct1:
                            st.subheader("Left : DCT for AI Reconstruction")
                            # Using use_container_width instead of use_column_width
                            st.image(dct_ai, caption="DSP-Aware Autoencoder Spectrum", use_container_width=True)
                            st.info("Note the smooth distribution of energy (Red/Yellow) fading naturally into Blue.")

                        with c_dct2:
                            st.subheader("Right : DCT for JPEG Construction")
                            # Using use_container_width instead of use_column_width
                            st.image(dct_jpeg, caption="Standard JPEG Spectrum", use_container_width=True)
                            st.warning("Note the 'Grid' patterns caused by 8x8 blocks and the sharp cutoff to Blue in corners.")