import os
import sys
import warnings
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import streamlit as st
import urllib.request

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="MediScan: AI-Powered Ocular Disease Diagnosis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .remedy-card {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        padding: 15px;
        border-radius: 6px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ocular_model():
    """Load VGG16 fine-tuned model using relative paths with Git LFS pointer detection."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "model.h5"),
        os.path.join(os.path.dirname(__file__), "model.h5"),
        "model.h5",
    ]
    
    model_path = None
    for path in possible_paths:
        abs_p = os.path.abspath(path)
        if os.path.exists(abs_p):
            model_path = abs_p
            break
            
    if model_path is None:
        st.error("❌ Model file `model.h5` not found. Please ensure `model.h5` is in the project root directory.")
        return None
        
    # Check if file is a Git LFS pointer text file (~130 bytes) instead of real 285MB model binary
    file_size = os.path.getsize(model_path)
    if file_size < 1000 * 1000: # Less than 1 MB
        st.warning("⚠️ **Git LFS Pointer File Detected**")
        st.error(
            f"The file `model.h5` at `{model_path}` is only **{file_size} bytes**. "
            "Render cloned the Git LFS pointer text file instead of downloading the 285 MB model binary."
        )
        st.info(
            "### How to Fix on Render:\n"
            "1. **Option A (Render Build Command)**: Set your Render Build Command to:\n"
            "   `git lfs install && git lfs pull && pip install -r requirements.txt`\n\n"
            "2. **Option B (Environment Variable)**: Set an Environment Variable in Render:\n"
            "   `MODEL_URL = <direct link to model.h5>`\n"
            "   The app will automatically download `model.h5` on boot."
        )
        
        # Check if MODEL_URL environment variable is provided for auto-download
        model_url = os.environ.get("MODEL_URL")
        if model_url:
            try:
                st.info("⏬ Downloading full `model.h5` from environment URL...")
                urllib.request.urlretrieve(model_url, model_path)
                st.success("✅ Model downloaded successfully!")
            except Exception as dl_err:
                st.error(f"Failed to download model from MODEL_URL: {str(dl_err)}")
                return None
        else:
            return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def is_retinal_fundus(image):
    """
    Validates if an uploaded image is a valid retinal fundus scan.
    Prevents shape errors and invalid predictions on non-retinal photos.
    """
    try:
        img_rgb = image.convert('RGB')
        arr = np.array(img_rgb, dtype=np.float32)
        h, w, c = arr.shape
        
        if h < 50 or w < 50:
            return False, "Image resolution is too low."

        # Corner Darkness & Background Uniformity Check
        tl = arr[:max(1, int(h * 0.1)), :max(1, int(w * 0.1))]
        tr = arr[:max(1, int(h * 0.1)), max(0, int(w * 0.9)):]
        bl = arr[max(0, int(h * 0.9)):, :max(1, int(w * 0.1))]
        br = arr[max(0, int(h * 0.9)):, max(0, int(w * 0.9)):]

        corners = np.concatenate([tl.reshape(-1, 3), tr.reshape(-1, 3), bl.reshape(-1, 3), br.reshape(-1, 3)], axis=0)
        corners_mean = float(corners.mean())
        corners_std = float(corners.std())

        if corners_mean > 135:
            return False, "Uploaded image does not have the dark background characteristic of a fundus camera scan."
        if corners_std > 65:
            return False, "Uploaded image background lacks dark uniformity."

        # Central Retinal Tissue Spectrum Check
        center = arr[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
        r_mean = float(center[:, :, 0].mean())
        g_mean = float(center[:, :, 1].mean())
        b_mean = float(center[:, :, 2].mean())

        if r_mean < 5 and g_mean < 5 and b_mean < 5:
            return False, "Uploaded image is blank or completely dark."

        if not (r_mean > b_mean * 0.95 or (r_mean >= g_mean * 0.75)):
            return False, "Color distribution does not match retinal tissue spectrum."

        return True, "Valid Retinal Fundus Scan"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def predict_image(image, model):
    """Preprocess image and perform prediction with VGG16 model."""
    size = (224, 224)
    image_rgb = image.convert('RGB')
    image_resized = ImageOps.fit(image_rgb, size, Image.Resampling.LANCZOS)
    
    img_array = np.asarray(image_resized, dtype=np.float32)
    img_reshape = img_array[np.newaxis, ...]
    
    predictions = model.predict(img_reshape, verbose=0)
    return predictions[0]


# Sidebar Content
with st.sidebar:
    eye_img_path = os.path.join(os.path.dirname(__file__), "eyejpg.jpg")
    if os.path.exists(eye_img_path):
        st.image(eye_img_path, width=280)
    else:
        st.title("👁️ MediScan")

    st.title("Ocular Disease Diagnostics")
    st.info(
        "**MediScan** leverages deep learning (VGG16 architecture) to classify "
        "retinal fundus scans into four categories:\n"
        "- Cataract\n"
        "- Diabetic Retinopathy\n"
        "- Glaucoma\n"
        "- Normal Healthy Eye"
    )
    st.markdown("---")
    st.caption("Developed with TensorFlow & Streamlit")

# Main Header
st.markdown('<div class="main-title">👁️ MEDI-SCAN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Medical Image Analysis for Ocular Disease Diagnosis</div>', unsafe_allow_html=True)

st.write(
    "Upload a **retinal fundus scan** below. The system will analyze the scan, detect potential "
    "ocular conditions, provide confidence scores, and suggest medical remedies."
)

# Load VGG16 Model
with st.spinner("Loading VGG16 deep learning model..."):
    model = load_ocular_model()

# File Uploader
uploaded_file = st.file_uploader(
    "Choose a retinal fundus image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("👆 Please upload an image file (JPG or PNG) to begin diagnosis.")
else:
    try:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Uploaded Scan", width=380)
            
        with col2:
            st.subheader("Diagnostic Results")
            
            # Step 1: Validate if image is a retinal fundus scan
            is_valid_fundus, msg = is_retinal_fundus(image)
            
            if not is_valid_fundus:
                st.error("⚠️ **Non-Retinal Image Error Detected**")
                st.warning(
                    f"The uploaded image was rejected as a non-retinal scan.\n\n"
                    f"**Reason**: {msg}\n\n"
                    "Please upload a clear retinal fundus photograph taken with an ophthalmic camera."
                )
            elif model is None:
                st.error("Model is not loaded. Cannot perform prediction.")
            else:
                with st.spinner("Analyzing retinal scan with VGG16 model..."):
                    probs = predict_image(image, model)
                    
                class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
                max_idx = int(np.argmax(probs))
                detected_class = class_names[max_idx]
                confidence = float(probs[max_idx]) * 100.0
                
                # Check for low model confidence
                if confidence < 35.0:
                    st.warning(
                        "⚠️ **Low Prediction Confidence**: The model could not confidently match "
                        "this scan with known ocular conditions. Please verify image quality."
                    )
                else:
                    st.markdown(f"### Predicted Condition: **{detected_class}**")
                    st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
                    
                    # Display Class Probabilities
                    st.markdown("#### Probability Breakdown:")
                    for idx, cls in enumerate(class_names):
                        st.write(f"**{cls}**: {probs[idx]*100:.1f}%")
                        st.progress(float(probs[idx]))
                    
                    # Diagnostics & Remedies
                    st.markdown("---")
                    if detected_class == 'Normal':
                        st.balloons()
                        st.success("✅ **Healthy Eye Detected**: No signs of Cataract, Glaucoma, or Diabetic Retinopathy were detected.")
                    elif detected_class == 'Cataract':
                        st.warning("⚠️ **Cataract Detected**")
                        st.markdown("#### 💊 Treatment & Remedy:")
                        st.info(
                            "Surgery is the primary effective treatment for cataracts to restore clear vision. "
                            "It involves removing the cloudy lens and replacing it with an artificial intraocular lens (IOL). "
                            "Consult an ophthalmologist for a comprehensive evaluation."
                        )
                    elif detected_class == 'Glaucoma':
                        st.warning("⚠️ **Glaucoma Detected**")
                        st.markdown("#### 💊 Treatment & Remedy:")
                        st.info(
                            "Prescription eye drops are the primary treatment for glaucoma to lower intraocular pressure. "
                            "Laser treatment and surgery may also be recommended depending on severity. "
                            "Immediate consultation with an eye care specialist is strongly advised to prevent vision loss."
                        )
                    elif detected_class == 'Diabetic Retinopathy':
                        st.warning("⚠️ **Diabetic Retinopathy Detected**")
                        st.markdown("#### 💊 Treatment & Remedy:")
                        st.info(
                            "Treatment options include anti-VEGF injections, corticosteroid medication, and laser photocoagulation "
                            "to reduce retinal swelling and seal leaking blood vessels. Strict blood sugar control is crucial."
                        )

    except Exception as e:
        st.error(f"❌ An error occurred while processing the image: {str(e)}")