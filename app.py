# ============================================================
# MALARIA DETECTION SYSTEM - WEB INTERFACE WITH SEVERITY
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Page configuration
st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🔬",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('malaria_cnn_model.h5')
    return model

model = load_model()

# ============================================================
# SEVERITY ANALYSIS FUNCTION
# ============================================================
def analyze_severity(img_array):
    """
    Analyze malaria severity based on parasite density
    """
    # Convert to HSV for color analysis
    img_uint8 = (img_array * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    
    # Define range for purple/dark spots (parasites)
    lower_purple = np.array([120, 30, 30])
    upper_purple = np.array([160, 255, 200])
    
    # Create mask for parasite regions
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Calculate parasite density
    parasite_density = (np.sum(mask > 0) / mask.size) * 100
    
    # Classify severity
    if parasite_density < 2:
        severity = "LOW PARASITEMIA"
        stage = "Early Stage"
        risk_level = "Low"
        color = "green"
        recommendation = "Start antimalarial treatment. Monitor symptoms."
    elif parasite_density < 5:
        severity = "MODERATE PARASITEMIA"
        stage = "Intermediate Stage"
        risk_level = "Medium"
        color = "orange"
        recommendation = "Immediate treatment required. Close monitoring advised."
    else:
        severity = "HIGH PARASITEMIA"
        stage = "Severe/Late Stage"
        risk_level = "High"
        color = "red"
        recommendation = "URGENT: Immediate medical attention required!"
    
    return {
        'parasite_density': parasite_density,
        'severity': severity,
        'stage': stage,
        'risk_level': risk_level,
        'color': color,
        'recommendation': recommendation
    }

# ============================================================
# APP HEADER
# ============================================================
st.title("🔬 Malaria Detection System")
st.markdown("**AI-Powered Blood Cell Analysis with Severity Classification**")
st.write("---")

# ============================================================
# FILE UPLOADER
# ============================================================
uploaded_file = st.file_uploader(
    "Upload a blood cell image", 
    type=['png', 'jpg', 'jpeg'],
    help="Upload a microscopic image of a blood cell"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Blood Cell', use_container_width=True)
    
    # Preprocess image
    img = np.array(image)
    
    # Handle different image formats
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img_resized = cv2.resize(img, (64, 64))
    img_normalized = img_resized / 255.0
    img_input = img_normalized.reshape(1, 64, 64, 3)
    
    with col2:
        if st.button('🔍 Analyze Image', use_container_width=True, type="primary"):
            with st.spinner('Analyzing blood cell...'):
                # Get model prediction
                prediction = model.predict(img_input, verbose=0)[0][0]
                confidence = float(prediction * 100) if prediction > 0.5 else float((1 - prediction) * 100)
                
                st.write("---")
                
                if prediction < 0.5:
                    # UNINFECTED
                    st.success("## ✅ UNINFECTED")
                    st.write("**Result:** No malaria parasite detected")
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    st.progress(min(confidence / 100, 1.0))
                    
                    st.info("**Status:** Healthy blood cell. No treatment required.")
                    
                else:
                    # INFECTED - Get severity
                    severity_info = analyze_severity(img_normalized)
                    
                    st.error("## ⚠️ PARASITIZED")
                    st.write("**Result:** Malaria parasite detected!")
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    st.progress(min(confidence / 100, 1.0))
                    
                    st.write("---")
                    st.subheader("📊 Severity Analysis (Exploratory)")
                    st.caption("Colour-based index, not clinically validated. "
                               "It is affected by stain colour and must not guide treatment.")
                    
                    # Severity metrics
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.metric(
                            label="Parasite Density",
                            value=f"{severity_info['parasite_density']:.2f}%"
                        )
                    
                    with metric_col2:
                        st.metric(
                            label="Risk Level",
                            value=severity_info['risk_level']
                        )
                    
                    # Severity classification with color
                    if severity_info['color'] == 'green':
                        st.success(f"**Stage:** {severity_info['stage']} ({severity_info['severity']})")
                    elif severity_info['color'] == 'orange':
                        st.warning(f"**Stage:** {severity_info['stage']} ({severity_info['severity']})")
                    else:
                        st.error(f"**Stage:** {severity_info['stage']} ({severity_info['severity']})")
                    
                    # Recommendation
                    st.write("---")
                    st.subheader("💊 Recommendation")
                    st.info(severity_info['recommendation'])
                    
                    # Severity scale visualization
                    st.write("---")
                    st.subheader("📈 Severity Scale")
                    
                    severity_scale = """
                    | Stage | Density | Risk | Action |
                    |-------|---------|------|--------|
                    | 🟢 Early | < 2% | Low | Monitor & Treat |
                    | 🟡 Intermediate | 2-5% | Medium | Immediate Treatment |
                    | 🔴 Severe | > 5% | High | Urgent Care! |
                    """
                    st.markdown(severity_scale)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
**Malaria Detection System**

This AI-powered application uses a Convolutional Neural Network (CNN) 
to detect malaria parasites in blood cell images and assess infection severity.

**Key Features:**
   - Binary Classification (Infected/Healthy)
   - Exploratory Severity Index (not clinically validated)
   - For research and demonstration only
""")

st.sidebar.title("📊 Model Performance")
st.sidebar.success("""
**Test split (2,000 images)**
- **Accuracy:** 95.15%
- **Precision:** 97.18%
- **Recall:** 93.00%
- **F1-Score:** 95.04%
- **Specificity:** 97.30%
- **AUC:** 0.9858

**Unused images (17,558)**
- **Accuracy:** 93.97%
- **AUC:** 0.9798
""")

st.sidebar.title("🔬 Severity Levels (Exploratory)")
st.sidebar.markdown("""
| Level | Density |
|-------|---------|
| 🟢 Early | < 2% |
| 🟡 Moderate | 2-5% |
| 🔴 Severe | > 5% |
""")

st.sidebar.title("📖 How to Use")
st.sidebar.write("""
1. Upload a blood cell image
2. Click 'Analyze Image'
3. View detection result
4. Check severity level
5. Follow recommendations
""")

# ============================================================
# FOOTER
# ============================================================
st.write("---")
st.markdown("""
<div style='text-align: center'>
    
</div>
""", unsafe_allow_html=True)
