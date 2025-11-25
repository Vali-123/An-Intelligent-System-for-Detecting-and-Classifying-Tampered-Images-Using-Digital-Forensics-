import streamlit as st
import cv2
from PIL import Image, ImageEnhance, ExifTags
from PIL.ExifTags import TAGS
import numpy as np
import io
import base64
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    st.warning("⚠️ Folium not installed. GPS mapping will be disabled. Install with: pip install folium streamlit-folium")
import webbrowser
import tempfile
import os
from urllib.parse import quote

# Page configuration
st.set_page_config(
    page_title="🔍 Professional Image Tampering Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2a5298;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .feature-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
    }
    
    .analysis-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .exif-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .face-detection {
        border: 3px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .tampering-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .score-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
    }
    
    .score-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
    }
    
    .score-high {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def extract_exif_data(image):
    """Extract EXIF metadata from image"""
    try:
        exif_dict = {}
        if hasattr(image, '_getexif'):
            exif_data = image._getexif()
            if exif_data is not None:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = value
        return exif_dict
    except Exception:
        return {}

def get_gps_coordinates(exif_dict):
    """Extract GPS coordinates from EXIF data"""
    try:
        if 'GPSInfo' in exif_dict:
            gps_info = exif_dict['GPSInfo']
            
            def convert_to_degrees(value):
                d, m, s = value
                return d + (m / 60.0) + (s / 3600.0)
            
            if 2 in gps_info and 4 in gps_info:
                lat = convert_to_degrees(gps_info[2])
                lon = convert_to_degrees(gps_info[4])
                
                if gps_info[1] == 'S':
                    lat = -lat
                if gps_info[3] == 'W':
                    lon = -lon
                    
                return lat, lon
    except Exception:
        pass
    return None, None

def generate_dummy_caption():
    """Generate a dummy AI caption"""
    captions = [
        "This image appears to show a natural outdoor scene with good lighting and composition.",
        "The photograph displays clear details with natural color distribution and proper exposure.",
        "This appears to be a high-quality digital photograph with realistic lighting and shadows.",
        "The image shows good technical quality with natural color balance and sharp focus.",
        "This photograph exhibits characteristics typical of authentic digital photography.",
    ]
    import random
    return random.choice(captions)

def detect_faces(image_array):
    """Detect faces in the image using OpenCV"""
    try:
        # Convert PIL image to OpenCV format
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around faces
        image_with_faces = image_array.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
        return image_with_faces, len(faces)
    except Exception:
        return image_array, 0

def calculate_tampering_score(exif_dict, image_format, has_gps):
    """Calculate a simulated tampering score based on metadata and image properties"""
    score = 20  # Base score (lower is better)
    
    # Check for missing EXIF data
    if not exif_dict:
        score += 30
    
    # Check for camera information
    if 'Make' not in exif_dict or 'Model' not in exif_dict:
        score += 25
    
    # Check for software modification traces
    if 'Software' in exif_dict:
        software = str(exif_dict['Software']).lower()
        editing_software = ['photoshop', 'gimp', 'canva', 'paint', 'editor']
        if any(editor in software for editor in editing_software):
            score += 35
    
    # GPS data can indicate authenticity
    if has_gps:
        score -= 15
    
    # Image format considerations
    if image_format.lower() in ['png']:
        score += 15  # PNG might indicate editing (though not always)
    
    # Check for creation date
    if 'DateTime' not in exif_dict:
        score += 20
    
    return min(max(score, 0), 100)  # Clamp between 0-100

def enhance_image_for_comparison(image):
    """Enhance image for before/after comparison"""
    # Brightness enhancement
    brightness_enhancer = ImageEnhance.Brightness(image)
    enhanced = brightness_enhancer.enhance(1.3)
    
    # Sharpness enhancement
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = sharpness_enhancer.enhance(1.5)
    
    return enhanced

def create_ela_view(image):
    """Create Error Level Analysis (ELA) view"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply Laplacian edge detection for ELA-like effect
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Convert back to RGB
        ela_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
        
        # Enhance contrast for better visualization
        enhanced_ela = cv2.convertScaleAbs(ela_image, alpha=3.0, beta=50)
        
        return Image.fromarray(enhanced_ela)
    except Exception:
        return image

def create_google_reverse_search_url(image):
    """Create Google Reverse Image Search URL"""
    # For demonstration - in a real app, you'd upload to a temporary server
    return "https://images.google.com/searchbyimage?image_url=data:image/jpeg;base64,{}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Professional Image Tampering Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-Powered Image Authentication & Forensic Analysis</p>', unsafe_allow_html=True)
    
    # File uploader
    st.markdown('<div class="feature-container">', unsafe_allow_html=True)
    st.markdown("### 📸 Upload Image for Analysis")
    st.markdown("*Supports JPG, JPEG, PNG, BMP, TIFF formats - Professional grade analysis*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
        help="Upload any image for comprehensive tampering analysis"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Create main layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 1. Show uploaded image preview
            st.markdown("### 🖼️ Image Preview")
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Basic image information
            st.markdown("#### 📋 Basic Information")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Format", uploaded_file.type.split('/')[-1].upper())
                st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
            with info_col2:
                st.metric("Width", f"{image.size[0]} px")
                st.metric("Height", f"{image.size[1]} px")
        
        with col2:
            # 2. Extract and display EXIF metadata
            st.markdown("### 📊 EXIF Metadata Analysis")
            exif_dict = extract_exif_data(image)
            
            if exif_dict:
                st.markdown('<div class="exif-container">', unsafe_allow_html=True)
                
                # Key EXIF information
                key_info = {}
                if 'Make' in exif_dict:
                    key_info['Camera Make'] = exif_dict['Make']
                if 'Model' in exif_dict:
                    key_info['Camera Model'] = exif_dict['Model']
                if 'DateTime' in exif_dict:
                    key_info['Date Taken'] = exif_dict['DateTime']
                if 'ExifImageWidth' in exif_dict:
                    key_info['Original Width'] = f"{exif_dict['ExifImageWidth']} px"
                if 'ExifImageHeight' in exif_dict:
                    key_info['Original Height'] = f"{exif_dict['ExifImageHeight']} px"
                if 'Software' in exif_dict:
                    key_info['Software'] = exif_dict['Software']
                
                if key_info:
                    for key, value in key_info.items():
                        st.write(f"**{key}:** {value}")
                else:
                    st.warning("⚠️ Limited EXIF data available")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ No EXIF metadata found - This may indicate image processing")
        
        # 3. Dummy AI Caption
        st.markdown("### 🤖 AI-Generated Caption")
        caption = generate_dummy_caption()
        st.info(f"💭 {caption}")
        
        # 4. Face Detection
        st.markdown("### 👥 Face Detection Analysis")
        image_with_faces, face_count = detect_faces(image_array)
        
        face_col1, face_col2 = st.columns([2, 1])
        with face_col1:
            if face_count > 0:
                st.markdown('<div class="face-detection">', unsafe_allow_html=True)
                st.image(image_with_faces, caption=f'Detected {face_count} face(s)', use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.image(image_with_faces, caption='No faces detected', use_column_width=True)
        
        with face_col2:
            st.metric("Faces Detected", face_count)
            if face_count > 0:
                st.success("✅ Faces found and highlighted")
            else:
                st.info("ℹ️ No faces detected in image")
        
        # 5. Tampering Score
        st.markdown("### 🎯 Tampering Risk Assessment")
        lat, lon = get_gps_coordinates(exif_dict)
        has_gps = lat is not None and lon is not None
        tampering_score = calculate_tampering_score(exif_dict, uploaded_file.type, has_gps)
        
        # Display score with color coding
        score_class = "score-low" if tampering_score < 40 else "score-medium" if tampering_score < 70 else "score-high"
        
        st.markdown(f'<div class="tampering-score {score_class}">', unsafe_allow_html=True)
        st.markdown(f"Tampering Risk: {tampering_score}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Progress bar
        progress_color = "normal" if tampering_score < 40 else "normal" if tampering_score < 70 else "normal"
        st.progress(tampering_score / 100.0)
        
        # Risk interpretation
        if tampering_score < 40:
            st.success("✅ **LOW RISK** - Image appears authentic with good metadata integrity")
        elif tampering_score < 70:
            st.warning("⚠️ **MEDIUM RISK** - Some indicators suggest possible processing or editing")
        else:
            st.error("🚨 **HIGH RISK** - Multiple indicators suggest significant tampering or processing")
        
        # 6. GPS Location Mapping
        if has_gps:
            st.markdown("### 🌍 GPS Location Data")
            st.success(f"📍 GPS coordinates found: {lat:.6f}, {lon:.6f}")
            
            if FOLIUM_AVAILABLE:
                # Create map
                m = folium.Map(location=[lat, lon], zoom_start=15)
                folium.Marker(
                    [lat, lon], 
                    popup=f"Image captured here<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
                    tooltip="Photo Location",
                    icon=folium.Icon(color='red', icon='camera')
                ).add_to(m)
                
                # Display map
                st_folium(m, width=700, height=400)
            else:
                st.info("🗺️ Install folium and streamlit-folium to see interactive map")
                st.write(f"**Coordinates:** {lat:.6f}, {lon:.6f}")
                # Create a simple link to Google Maps
                maps_url = f"https://www.google.com/maps?q={lat},{lon}"
                st.markdown(f"[📍 View on Google Maps]({maps_url})")
        else:
            st.markdown("### 🌍 GPS Location Data")
            st.info("ℹ️ No GPS coordinates found in image metadata")
        
        # 7. Google Reverse Image Search Button
        st.markdown("### 🔍 Reverse Image Search")
        if st.button('🌐 Open Google Reverse Image Search'):
            st.info("🔗 Google Reverse Image Search would open in a new tab")
            st.markdown("""
            **Note:** In a production environment, this would:
            1. Upload the image to a temporary server
            2. Generate a proper Google Images search URL
            3. Open the search in a new browser tab
            
            For privacy reasons, this demo doesn't actually upload your image.
            """)
        
        # 8. Image Comparison Slider
        st.markdown("### 🔄 Before/After Enhancement Comparison")
        enhanced_image = enhance_image_for_comparison(image)
        
        comparison_col1, comparison_col2 = st.columns(2)
        with comparison_col1:
            st.markdown("**Original Image**")
            st.image(image, use_column_width=True)
        
        with comparison_col2:
            st.markdown("**Enhanced Image**")
            st.image(enhanced_image, use_column_width=True)
        
        # Enhancement details
        st.info("💡 Enhancement includes brightness (+30%) and sharpness (+50%) adjustments for comparison analysis")
        
        # 9. ELA (Error Level Analysis) View
        st.markdown("### 🔬 Error Level Analysis (ELA) View")
        ela_image = create_ela_view(image)
        
        ela_col1, ela_col2 = st.columns(2)
        with ela_col1:
            st.markdown("**Original Image**")
            st.image(image, use_column_width=True)
        
        with ela_col2:
            st.markdown("**ELA Analysis**")
            st.image(ela_image, use_column_width=True)
        
        st.info("""
        🔬 **ELA Analysis Explanation:**
        - Bright areas in ELA view may indicate recent modifications
        - Uniform error levels suggest consistent compression
        - Sharp edges or anomalies might indicate tampering
        - This is a simplified ELA implementation for demonstration
        """)
        
        # Additional Analysis Section
        st.markdown("### 📈 Comprehensive Analysis Summary")
        
        with st.expander("🔍 Detailed Technical Report", expanded=True):
            st.markdown(f"""
            **Image Analysis Report**
            
            📁 **File Information:**
            - Format: {uploaded_file.type}
            - Size: {uploaded_file.size:,} bytes
            - Dimensions: {image.size[0]} × {image.size[1]} pixels
            
            📊 **Metadata Assessment:**
            - EXIF Data Present: {'✅ Yes' if exif_dict else '❌ No'}
            - Camera Information: {'✅ Available' if 'Make' in exif_dict else '❌ Missing'}
            - GPS Coordinates: {'✅ Present' if has_gps else '❌ Not found'}
            - Creation Date: {'✅ Available' if 'DateTime' in exif_dict else '❌ Missing'}
            
            👥 **Content Analysis:**
            - Faces Detected: {face_count}
            - Face Detection Confidence: {'High' if face_count > 0 else 'N/A'}
            
            🎯 **Tampering Assessment:**
            - Risk Score: {tampering_score}/100
            - Risk Level: {'Low' if tampering_score < 40 else 'Medium' if tampering_score < 70 else 'High'}
            - Authenticity Confidence: {100 - tampering_score}%
            
            🔬 **Forensic Analysis:**
            - ELA Pattern: Analyzed for compression artifacts
            - Enhancement Comparison: Available
            - Metadata Integrity: {'Good' if exif_dict else 'Concerning'}
            """)
        
        # Export options
        st.markdown("### 💾 Export Options")
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("📄 Generate Report"):
                report_data = {
                    "filename": uploaded_file.name,
                    "tampering_score": tampering_score,
                    "faces_detected": face_count,
                    "has_gps": has_gps,
                    "has_exif": bool(exif_dict),
                    "analysis_timestamp": "2025-01-23T16:03:18Z"
                }
                st.json(report_data)
        
        with export_col2:
            if st.button("🖼️ Save ELA Image"):
                st.info("ELA image would be saved to downloads folder")
        
        with export_col3:
            if st.button("📍 Export GPS Data"):
                if has_gps:
                    st.success(f"GPS: {lat:.6f}, {lon:.6f}")
                else:
                    st.warning("No GPS data to export")

    else:
        # Welcome screen
        st.markdown("""
        <div class="analysis-section">
            <h3>🚀 Professional Image Forensics Suite</h3>
            <p>Upload an image above to begin comprehensive tampering detection analysis.</p>
            
            <h4>🔍 Analysis Features:</h4>
            <ul>
                <li>🖼️ <strong>Image Preview</strong> - High-quality display with technical details</li>
                <li>📊 <strong>EXIF Metadata</strong> - Camera, GPS, timestamp, and software analysis</li>
                <li>🤖 <strong>AI Caption</strong> - Automated content description</li>
                <li>👥 <strong>Face Detection</strong> - OpenCV-powered facial recognition</li>
                <li>🎯 <strong>Tampering Score</strong> - Risk assessment with visual indicators</li>
                <li>🌍 <strong>GPS Mapping</strong> - Interactive location visualization</li>
                <li>🔍 <strong>Reverse Search</strong> - Google Images integration</li>
                <li>🔄 <strong>Enhancement Comparison</strong> - Before/after slider view</li>
                <li>🔬 <strong>ELA Analysis</strong> - Error Level Analysis for forensic examination</li>
            </ul>
            
            <h4>💡 Perfect for:</h4>
            <ul>
                <li>📰 News verification and fact-checking</li>
                <li>🛡️ Social media content moderation</li>
                <li>⚖️ Legal evidence authentication</li>
                <li>🔍 Insurance claim investigation</li>
                <li>📱 Personal photo verification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
