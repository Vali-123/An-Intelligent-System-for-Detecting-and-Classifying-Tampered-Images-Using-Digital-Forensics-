import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import io
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our detection system
from analyze_single_image import SingleImageTamperingDetector

# Configure Streamlit page
st.set_page_config(
    page_title="AI Image Tampering Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .danger-card {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_confidence_gauge(confidence, title):
    """Create a gauge chart for confidence visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_method_comparison_chart(analysis_data):
    """Create a bar chart comparing different detection methods"""
    methods = []
    confidences = []
    descriptions = []
    
    for method, data in analysis_data.items():
        method_name = method.replace('_', ' ').title()
        methods.append(method_name)
        confidences.append(data['confidence'] * 100)
        descriptions.append(data['description'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods,
            y=confidences,
            text=[f"{conf:.1f}%" for conf in confidences],
            textposition='auto',
            marker_color=['red' if conf > 70 else 'orange' if conf > 30 else 'green' for conf in confidences],
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Detection Method Comparison",
        xaxis_title="Detection Methods",
        yaxis_title="Confidence (%)",
        height=400,
        showlegend=False
    )
    
    fig.add_hline(y=30, line_dash="dash", line_color="orange", 
                  annotation_text="Low Threshold (30%)")
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="High Threshold (70%)")
    
    return fig

def display_analysis_results(results):
    """Display comprehensive analysis results"""
    
    # Overall Assessment Section
    st.markdown("## 📊 Analysis Results")
    
    confidence = results['overall_assessment']['tampering_confidence']
    likely_tampered = results['overall_assessment']['likely_tampered']
    severity = results['overall_assessment']['severity']
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Overall Confidence",
            value=f"{confidence:.3f}",
            delta=f"{confidence*100:.1f}%"
        )
    
    with col2:
        status_color = "🚨" if likely_tampered else "✅"
        st.metric(
            label="⚠️ Likely Tampered",
            value=f"{status_color} {'YES' if likely_tampered else 'NO'}"
        )
    
    with col3:
        severity_emoji = "🔴" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
        st.metric(
            label="📈 Severity Level",
            value=f"{severity_emoji} {severity}"
        )
    
    with col4:
        st.metric(
            label="📐 Image Size",
            value=f"{results['image_shape'][1]}×{results['image_shape'][0]}"
        )
    
    # Interpretation
    st.markdown("### 💭 AI Interpretation")
    interpretation = results['interpretation']
    
    if "HIGH likelihood" in interpretation:
        st.markdown(f'<div class="danger-card">{interpretation}</div>', unsafe_allow_html=True)
    elif "MEDIUM likelihood" in interpretation:
        st.markdown(f'<div class="warning-card">{interpretation}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="success-card">{interpretation}</div>', unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence gauge
        gauge_fig = create_confidence_gauge(confidence, "Overall Tampering Confidence")
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        # Method comparison chart
        method_fig = create_method_comparison_chart(results['analysis'])
        st.plotly_chart(method_fig, use_container_width=True)
    
    # Detailed Method Breakdown
    st.markdown("### 🔬 Detailed Method Breakdown")
    
    for method, data in results['analysis'].items():
        method_name = method.replace('_', ' ').title()
        conf = data['confidence']
        
        # Create expandable section for each method
        with st.expander(f"🔍 {method_name} - Confidence: {conf:.3f}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Description:** {data['description']}")
                
                # Show specific findings
                findings = []
                if 'matches' in data and data['matches'] > 0:
                    findings.append(f"📋 Found {data['matches']} suspicious matches")
                if 'outliers' in data and data['outliers'] > 0:
                    findings.append(f"🔊 Detected {data['outliers']} noise outliers")
                if 'suspicious_blocks' in data and data['suspicious_blocks'] > 0:
                    findings.append(f"📸 Found {data['suspicious_blocks']} suspicious compression blocks")
                if 'inconsistent_regions' in data and data['inconsistent_regions'] > 0:
                    findings.append(f"💡 Identified {data['inconsistent_regions']} inconsistent lighting regions")
                if 'suspicious_edges' in data and data['suspicious_edges'] > 0:
                    findings.append(f"🔍 Detected {data['suspicious_edges']} suspicious edge patterns")
                
                if findings:
                    st.write("**Findings:**")
                    for finding in findings:
                        st.write(f"• {finding}")
                else:
                    st.write("• ✅ No suspicious patterns detected by this method")
            
            with col2:
                # Mini gauge for this method
                mini_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = conf * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ]
                    }
                ))
                mini_gauge.update_layout(height=200)
                st.plotly_chart(mini_gauge, use_container_width=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔍 AI Image Tampering Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to detect potential tampering using advanced AI algorithms</p>', unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## 📋 System Information")
        st.info("""
        **Detection Methods:**
        🔍 Copy-Move Forgery Detection
        🔊 Noise Pattern Analysis  
        📸 JPEG Compression Artifacts
        💡 Lighting Consistency Analysis
        🔍 Edge Artifact Detection
        """)
        
        st.markdown("## 📈 Confidence Scale")
        st.success("**0.0 - 0.3:** Low likelihood (likely authentic)")
        st.warning("**0.3 - 0.7:** Medium likelihood (suspicious)")
        st.error("**0.7 - 1.0:** High likelihood (likely tampered)")
        
        st.markdown("## 📊 Supported Formats")
        st.write("• JPG, JPEG")
        st.write("• PNG")
        st.write("• BMP")
        st.write("• TIFF, TIF")
    
    # File upload section
    st.markdown("## 📁 Upload Image for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
        help="Upload an image file to analyze for potential tampering"
    )
    
    if uploaded_file is not None:
        # Display upload info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 File Name", uploaded_file.name)
        with col2:
            st.metric("📊 File Size", f"{uploaded_file.size:,} bytes")
        with col3:
            st.metric("🕐 Upload Time", datetime.now().strftime("%H:%M:%S"))
        
        # Show uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        
        with col2:
            st.markdown("### 📐 Image Properties")
            st.write(f"**Dimensions:** {image.width} × {image.height} pixels")
            st.write(f"**Mode:** {image.mode}")
            st.write(f"**Format:** {image.format}")
            if hasattr(image, 'info') and image.info:
                st.write(f"**Info:** {len(image.info)} metadata entries")
        
        # Analysis button
        if st.button("🚀 Analyze Image for Tampering", type="primary", use_container_width=True):
            
            # Show progress
            with st.spinner("🔍 Analyzing image... This may take a few moments..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Initialize detector
                    detector = SingleImageTamperingDetector()
                    
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Update progress
                    progress_bar.progress(20)
                    status_text.text("🔍 Loading image...")
                    
                    # Load image for analysis
                    image_array = np.array(image.convert('RGB'))
                    
                    progress_bar.progress(40)
                    status_text.text("🔍 Detecting copy-move forgery...")
                    
                    # Run copy-move analysis
                    copy_move_matches, cm_confidence = detector.detect_copy_move_forgery(image_array)
                    
                    progress_bar.progress(55)
                    status_text.text("🔊 Analyzing noise patterns...")
                    
                    # Run noise analysis
                    noise_outliers, noise_confidence = detector.analyze_noise_patterns(image_array)
                    
                    progress_bar.progress(70)
                    status_text.text("📸 Detecting JPEG artifacts...")
                    
                    # Run JPEG analysis
                    jpeg_artifacts, jpeg_confidence = detector.detect_jpeg_compression_artifacts(image_array)
                    
                    progress_bar.progress(85)
                    status_text.text("💡 Analyzing lighting consistency...")
                    
                    # Run lighting analysis
                    lighting_issues, lighting_confidence = detector.analyze_lighting_consistency(image_array)
                    
                    progress_bar.progress(95)
                    status_text.text("🔍 Detecting edge artifacts...")
                    
                    # Run edge analysis
                    edge_artifacts, edge_confidence = detector.detect_edge_artifacts(image_array)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Compile results
                    results = {
                        "image_path": uploaded_file.name,
                        "image_name": uploaded_file.name,
                        "image_shape": list(image_array.shape),
                        "analysis": {
                            "copy_move": {
                                "matches": len(copy_move_matches),
                                "confidence": float(cm_confidence),
                                "description": "Detects duplicated regions within the image"
                            },
                            "noise_analysis": {
                                "outliers": len(noise_outliers),
                                "confidence": float(noise_confidence),
                                "description": "Identifies inconsistent noise distributions"
                            },
                            "jpeg_artifacts": {
                                "suspicious_blocks": len(jpeg_artifacts),
                                "confidence": float(jpeg_confidence),
                                "description": "Analyzes compression inconsistencies"
                            },
                            "lighting": {
                                "inconsistent_regions": len(lighting_issues),
                                "confidence": float(lighting_confidence),
                                "description": "Detects unnatural lighting variations"
                            },
                            "edge_artifacts": {
                                "suspicious_edges": len(edge_artifacts),
                                "confidence": float(edge_confidence),
                                "description": "Identifies suspicious edge patterns from splicing"
                            }
                        }
                    }
                    
                    # Calculate overall confidence
                    confidences = [cm_confidence, noise_confidence, jpeg_confidence, 
                                 lighting_confidence, edge_confidence]
                    overall_confidence = np.mean(confidences)
                    
                    results["overall_assessment"] = {
                        "tampering_confidence": round(float(overall_confidence), 3),
                        "likely_tampered": bool(overall_confidence > 0.3),
                        "severity": "High" if overall_confidence > 0.7 else "Medium" if overall_confidence > 0.3 else "Low"
                    }
                    
                    # Add interpretation
                    if overall_confidence > 0.7:
                        interpretation = "🚨 HIGH likelihood of tampering detected! Multiple detection methods show strong evidence of manipulation."
                    elif overall_confidence > 0.3:
                        interpretation = "⚠️ MEDIUM likelihood of tampering detected. Some suspicious patterns found, requires closer inspection."
                    else:
                        interpretation = "✅ LOW likelihood of tampering. Image appears authentic or contains minimal suspicious patterns."
                    
                    results["interpretation"] = interpretation
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    display_analysis_results(results)
                    
                    # Download JSON results
                    st.markdown("### 💾 Download Results")
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label="📄 Download JSON Report",
                        data=json_str,
                        file_name=f"{uploaded_file.name.split('.')[0]}_tampering_analysis.json",
                        mime="application/json"
                    )
                    
                    # Clean up temp file
                    import os
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                except Exception as e:
                    st.error(f"❌ An error occurred during analysis: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
    
    else:
        # Show information when no file is uploaded
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🎯 How to Use This System:
        
        1. **📁 Upload an Image:** Click the file uploader above and select your image
        2. **🚀 Start Analysis:** Click the "Analyze Image" button
        3. **📊 View Results:** Get detailed tampering analysis with confidence scores
        4. **💾 Download Report:** Save the results as a JSON file
        
        ### 🔬 What We Detect:
        
        - **🔍 Copy-Move Forgery:** Duplicated regions within the same image
        - **🔊 Noise Inconsistencies:** Unnatural noise patterns indicating manipulation
        - **📸 JPEG Artifacts:** Compression inconsistencies from editing
        - **💡 Lighting Problems:** Inconsistent lighting that suggests splicing
        - **🔍 Edge Artifacts:** Suspicious edges from image manipulation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show sample results
        st.markdown("### 📊 Sample Analysis Results")
        sample_data = {
            "Copy-Move Detection": 0.85,
            "Noise Analysis": 0.45,
            "JPEG Artifacts": 0.32,
            "Lighting Analysis": 0.18,
            "Edge Artifacts": 0.25
        }
        
        sample_df = pd.DataFrame(list(sample_data.items()), columns=['Method', 'Confidence'])
        st.bar_chart(sample_df.set_index('Method'))

if __name__ == "__main__":
    main()
