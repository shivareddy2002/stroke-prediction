import streamlit as st
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import time

# ================== PAGE CONFIG ===================
st.set_page_config(
    page_title="Scan4Stroke",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== ASSET LOADING ===================
MODEL_PATH = "GA_BiGRU_Improved.h5"
CHROMOSOME_PATH = "GA_BiGRU_best_chromosome.npy"
BG_IMAGE_PATH = "bg_image.jpeg.jpg"
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image_base64 = get_base64_image(BG_IMAGE_PATH)

@st.cache_resource
def load_assets():
    try:
        model = load_model(MODEL_PATH)
        best_chromosome = np.load(CHROMOSOME_PATH)
        selected_features = np.flatnonzero(best_chromosome)
        return model, selected_features
    except Exception:
        return None, None

model, selected_features = load_assets()

# ================== CSS STYLING ===================
def load_css():
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_image_base64}");
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        min-height: 100vh;
        width: 100vw;
    }}
    .navbar {{
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 9999;
        height: 70px;
        background: rgba(255, 255, 255, 0.96);
        box-shadow: 0 2px 12px rgba(0,0,0,0.09);
        display: flex;
        align-items: center;
        padding: 0 2rem;
        font-family: "Segoe UI", sans-serif;
    }}
    .navbar-logo {{
        font-size: 1.8rem;
        font-weight: bold;
        color: #007bff;
    }}
    .navbar-tagline {{
        margin-left: 1.2rem;
        font-size: 1.1rem;
        color: #495464;
        font-weight: 500;
    }}
    .main .block-container {{
        padding-top: 85px !important;
    }}
    .hero-container {{
        max-width: 700px;
        padding: 2.4rem 1rem 3.5rem 0;
        color: white;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.78);
    }}
    .big-title {{
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }}
    .subtitle {{
        font-size: 2.1rem;
        color: #7ec8ff;
        margin-bottom: 0.9rem;
        font-weight: 600;
    }}
    .desc-text {{
        font-size: 1.15rem;
        margin-bottom: 2.3rem;
        opacity: 0.96;
    }}
    .action-btn {{
        background-color: #147EFB;
        color: white;
        border: none;
        border-radius: 0.7rem;
        font-size: 1.15rem;
        font-weight: 600;
        padding: 0.78rem 2.1rem;
        margin-bottom: 7px;
        margin-right: 10px;
        cursor: pointer;
        transition: background 0.2s;
    }}
    .action-btn:hover {{
        background: #0056b3;
        box-shadow: 0 4px 24px #81cfff55;
    }}
    @media (max-width: 900px) {{
        .big-title {{ font-size: 2.1rem; }}
        .hero-container {{ padding: 1.1rem 0.5rem 2rem 0; }}
    }}
    </style>
    """, unsafe_allow_html=True)

# ================== COMPONENTS ===================
def navbar():
    st.markdown("""
    <div class="navbar">
        <span class="navbar-logo">Scan4Stroke</span>
        <span class="navbar-tagline">Quick stroke detection from CT/MRI scans</span>
    </div>
    """, unsafe_allow_html=True)

def hero_section():
    st.markdown("""
    <div class="hero-container">
        <div class="big-title">We Care for Your Health Every Moment</div>
        <div class="subtitle">Advanced Stroke Prediction System</div>
        <div class="desc-text">
            Upload CT or MRI scans to receive instant stroke risk predictions using state-of-the-art deep learning technology.
        </div>
        <a href="#upload-scan"><button class="action-btn">Get Started</button></a>
    </div>
    """, unsafe_allow_html=True)

def sidebar_details():
    st.sidebar.markdown("### 📋 About the App")
    st.sidebar.info("""
**Scan4Stroke**  
**Model:** GA-BiGRU(Genetic Algorithm-optimized Bidirectional GRU)  
**Input:** CT/MRI Brain Scans  
**Output:** Stroke vs Normal Classification


Features:
- AI-powered deep learning
- Genetic algorithm optimization
- Fast and reliable
""")


# ================== IMAGE PREPROCESSING ===================
def preprocess_image(img_file, image_size=(128, 128), selected_features=None):
    try:
        img = Image.open(img_file).convert('L').resize(image_size)
        img_array = np.array(img) / 255.0
        flat_img = img_array.flatten()
        if selected_features is not None:
            flat_img = flat_img[selected_features]
        return flat_img.reshape(1, 1, -1)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# ================== PREDICTION & DISPLAY ===================
def predict_and_display(uploaded_file, model, selected_features):
    with st.spinner("Analyzing the scan with AI..."):
        start_time = time.time()
        processed_img = preprocess_image(uploaded_file, selected_features=selected_features)
        if processed_img is not None:
            try:
                pred_prob = model.predict(processed_img, verbose=0)[0][0]
                processing_time = time.time() - start_time
                pred_class = "Stroke" if pred_prob > 0.5 else "Normal"
                confidence = pred_prob * 100 if pred_prob > 0.5 else (1 - pred_prob) * 100

                col_left, col_center, col_right = st.columns([1,2,1])
                with col_center:
                    st.image(uploaded_file, caption="📸 Analyzed Scan", use_container_width=True)

                result_color = "#d7263d" if pred_class == "Stroke" else "#248232"
                bg_card = "rgba(44,19,29,0.72)" if pred_class == "Stroke" else "rgba(23,44,25,0.80)"

                st.markdown(
                    f"""
                    <div style='
                        font-size:2.0rem;
                        font-weight: bold;
                        color:{result_color};
                        padding: 1.3rem 2.2rem 1.1rem 2.2rem;
                        background:{bg_card};
                        border-radius:1.1rem;
                        margin: 1.4rem auto 0.6rem auto;
                        width:fit-content;
                        text-align:center;
                        box-shadow: 0 4px 28px #13336650,0 2px 6px #391b1b60;
                        text-shadow: 2px 2px 12px #fff, 0 2px 40px #222334cc;
                    '>
                      <b>
                        { '🚨 Stroke Detected' if pred_class == 'Stroke' else '✅ Normal Scan' } ({confidence:.1f}%)
                      </b>
                    </div>
                    <div style='
                        font-size:1.2rem;
                        font-weight: bold;
                        color: #fff;
                        background:rgba(20,34,44,0.65);
                        border-radius:0.8rem;
                        margin: 0.8rem auto 1.2rem auto;
                        width:fit-content;
                        padding: 0.7rem 2.1rem;
                        box-shadow: 0 2px 12px #0003;
                        text-align:center;
                        text-shadow: 1px 1px 10px #000, 0 1px 8px #2e53afc5;
                    '>
                      <b>
                        Processing Time: {processing_time:.2f} s   |   Model: GA-BiGRU   |   Features: {len(selected_features)}   |   Resolution: 128x128
                      </b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"❌ **Prediction Error:** {str(e)}")
        else:
            st.error("❌ **Image Processing Failed:** Unable to process the uploaded image.")


# ================== APP MAIN ===================
def main():
    load_css()
    navbar()


    col1, col2 = st.columns([1,1])
    with col1:
        hero_section()
    with col2:
        st.empty()  # place holder for features or empty


    st.markdown("<hr>", unsafe_allow_html=True)


    # Upload Anchor + Smooth-Scroll JS
    st.markdown('<a id="upload-scan"></a>', unsafe_allow_html=True)
    st.markdown("""
    <script>
    window.addEventListener('load', function() {
        if (window.location.hash === "#upload-scan") {
            setTimeout(function() {
                const el = document.getElementById('upload-scan');
                if (el) {
                    el.scrollIntoView({behavior: 'smooth', block: 'start'});
                }
            }, 100);
        }
    });
    </script>
    """, unsafe_allow_html=True)


    if model is None or selected_features is None:
        st.error("⚠️ **Model files not found!** Please ensure model and feature files are in the app directory.")
        st.info("📁 **Required files:**\n- `GA_BiGRU_Improved.h5` (trained model)\n- `GA_BiGRU_best_chromosome.npy` (feature selection)")
        st.stop()


    st.markdown("### 📤 Upload Medical Scan")
    st.markdown("Upload a CT or MRI brain scan image for AI analysis. Supported formats: JPG, JPEG, PNG")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Select a clear CT or MRI brain scan image for best results")


    if uploaded_file is not None:
        st.markdown("#### 🖼️ Uploaded Scan Preview")
        predict_and_display(uploaded_file, model, selected_features)


    sidebar_details()


if __name__ == "__main__":
    main()
