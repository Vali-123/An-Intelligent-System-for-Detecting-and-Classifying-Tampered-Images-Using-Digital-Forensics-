import streamlit as st
from PIL import Image, ExifTags
import os
import json
import tempfile
import cv2
import numpy as np
from quality_based_detector import QualityBasedTamperingDetector

# Initialize the detector
detector = QualityBasedTamperingDetector()

# Custom JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Custom CSS styling
def apply_custom_styles():
    st.markdown("""
    <style>
    .stApp { background-color: #f9f9fb; }
    [data-testid="stSidebar"] {
        background-color: #1e272e !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stDownloadButton>button {
        background-color: #2ecc71 !important;
        color: white;
    }
    [data-testid="column"] {
        background-color: white;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Enhancements
def enhance_sharpness(image, factor):
    kernel = np.array([[-1,-1,-1],[-1,9+factor,-1],[-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def reduce_noise(image, strength):
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

def adjust_contrast(image, factor):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def safe_delete_temp_file(path):
    try:
        if os.path.exists(path): os.unlink(path)
    except: pass

def extract_metadata(image):
    try:
        exif = image._getexif()
        if not exif: return {}
        return {ExifTags.TAGS.get(tag): val for tag, val in exif.items() if tag in ExifTags.TAGS}
    except:
        return {}

def display_analysis_results(results):
    with st.expander("📊 Summary", expanded=True):
        col1, col2 = st.columns([1,3])
        with col1:
            prob = results['tampering_assessment']['tampering_probability']
            verdict = results['tampering_assessment']['verdict']
            quality = results['quality_metrics']['overall_quality_score']
            st.metric("Tampering Probability", f"{prob:.1%}")
            st.metric("Quality Score", f"{quality:.2f}/1.00")
            if "Not Tampered" in verdict:
                st.success(verdict)
            elif "Likely Tampered" in verdict:
                st.error(verdict)
            else:
                st.warning(verdict)
        with col2:
            analysis = results['quality_analysis']
            issues = []
            if analysis['is_blurry']: issues.append("🔍 **Blurry**")
            if analysis['is_noisy']: issues.append("📢 **Noisy**")
            if analysis['is_low_resolution']: issues.append("📉 **Low resolution**")
            if analysis['has_compression_artifacts']: issues.append("💾 **Compression artifacts**")
            if issues:
                st.markdown("### Quality Issues:")
                for i in issues: st.markdown(f"- {i}")
            else:
                st.success("✅ No major quality issues detected")

    with st.expander("🔬 Detailed Metrics"):
        m = results['quality_metrics']
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Blur Score", f"{m['blur_score']:.3f}")
            st.metric("Laplacian", f"{m['blur_value']:.1f}")
            st.metric("Noise Score", f"{m['noise_score']:.3f}")
            st.metric("Noise Level", f"{m['noise_value']:.1f}")
        with c2:
            st.metric("Sharpness Score", f"{m['sharpness_score']:.3f}")
            st.metric("Gradient", f"{m['sharpness_value']:.1f}")
            st.metric("Compression Score", f"{m['compression_score']:.3f}")
            st.metric("Artifacts", f"{m['compression_artifacts']:.1f}")

    with st.expander("💡 Recommendations"):
        prob = results['tampering_assessment']['tampering_probability']
        if prob > 0.7:
            st.error("🚨 High Risk: Likely Tampered")
        elif prob > 0.4:
            st.warning("⚠️ Medium Risk: Check Carefully")
        else:
            st.success("✅ Low Risk: Looks Authentic")
        st.markdown("""
        **Tips:**
        - Use images with >1280x720 resolution
        - Avoid excessive compression
        - Prefer high sharpness and low noise
        - Use reputable image sources
        """)

    try:
        json_data = json.dumps(results, indent=2, cls=NumpyEncoder)
        st.download_button("📥 Download Report", json_data, "image_report.json", "application/json")
    except:
        st.warning("Could not export report")

# Main app
def main():
    st.set_page_config(page_title="Image Forensic Analyzer", page_icon="📷", layout="wide")
    apply_custom_styles()

    st.sidebar.title("🧪 Image Forensic Analyzer")
    st.sidebar.markdown("Detect tampering & enhance image quality.")

    uploaded_file = st.sidebar.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    app_mode = st.sidebar.radio("Choose Mode", ["🔍 Tampering Detection", "✨ Image Enhancement"])
    
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    st.title("🔬 Image Forensic Analyzer")
    st.markdown("Detect tampered images based on quality metrics & improve them.")

    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        st.session_state.original_image = image_cv

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original Image")
            st.image(image_pil, use_column_width=True)
        with c2:
            st.subheader("🧾 Image Metadata")
            meta = extract_metadata(image_pil)
            if meta:
                for k, v in meta.items():
                    st.markdown(f"**{k}**: {v}")
            else:
                st.info("No EXIF metadata found.")

        if app_mode == "🔍 Tampering Detection":
            if st.button("🔍 Analyze Tampering"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image_pil.save(tmp.name)
                    tmp_path = tmp.name
                try:
                    result = detector.analyze_image_quality(tmp_path)
                    result = json.loads(json.dumps(result, cls=NumpyEncoder))
                    st.session_state.analysis_results = result
                    display_analysis_results(result)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                finally:
                    safe_delete_temp_file(tmp_path)
            elif st.session_state.analysis_results:
                display_analysis_results(st.session_state.analysis_results)

        elif app_mode == "✨ Image Enhancement":
            st.sidebar.subheader("Enhancement Controls")
            sharpness = st.sidebar.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
            brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
            noise = st.sidebar.slider("Noise Reduction", 0, 30, 0, 1)

            if st.sidebar.button("✨ Apply Enhancements"):
                enhanced = image_cv.copy()
                if sharpness != 1.0: enhanced = enhance_sharpness(enhanced, sharpness)
                if brightness != 1.0: enhanced = adjust_brightness(enhanced, brightness)
                if contrast != 1.0: enhanced = adjust_contrast(enhanced, contrast)
                if noise > 0: enhanced = reduce_noise(enhanced, noise)
                st.session_state.current_image = enhanced

            if st.session_state.current_image is not None:
                st.subheader("Enhanced Image")
                enhanced_rgb = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB)
                enhanced_pil = Image.fromarray(enhanced_rgb)
                st.image(enhanced_pil, use_column_width=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    enhanced_pil.save(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button("📥 Download Enhanced Image", f, "enhanced.jpg", "image/jpeg")
                    safe_delete_temp_file(tmp.name)
            else:
                st.info("Adjust sliders & apply enhancements")

    else:
        st.info("📤 Upload an image to begin.")

if __name__ == "__main__":
    main()
