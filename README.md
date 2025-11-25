# AI-Based Image Tampering Detection System

## Overview
This project implements a comprehensive AI-based system for detecting image tampering and manipulation. It uses multiple computer vision and image processing techniques to identify various types of forgeries including copy-move, splicing, noise inconsistencies, lighting artifacts, and compression anomalies.

## Features

### Detection Methods
1. **Copy-Move Forgery Detection**: Identifies duplicated regions within the same image
2. **Noise Pattern Analysis**: Detects inconsistent noise distributions
3. **JPEG Compression Artifacts**: Analyzes compression inconsistencies
4. **Lighting Consistency Analysis**: Detects unnatural lighting variations
5. **Edge Artifact Detection**: Identifies suspicious edge patterns from splicing

### Output
- **Confidence Scores**: Numerical confidence for each detection method (0.0 - 1.0)
- **Visual Analysis**: Color-coded visualizations highlighting suspicious regions
- **Overall Assessment**: Combined analysis with severity rating
- **Detailed Reports**: JSON format with comprehensive analysis results

## Installation

### Requirements
- Python 3.7+
- Required packages (install via requirements.txt)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample test images (optional)
python generate_test_images.py

# Run the main detection system
python image_tampering_detector.py
```

## Usage

### Basic Usage
1. Place your image files (.jpg, .jpeg, .png, .bmp, .tiff) in the project directory
2. Run the main script: `python image_tampering_detector.py`
3. The system will automatically detect and analyze all images
4. Results will be displayed and saved to `tampering_analysis_results.json`

### Sample Images
The project includes a sample image generator that creates test images with different types of tampering:
- Authentic (clean) image
- Copy-move tampered image
- Spliced image with compression artifacts
- Noise-inconsistent image
- Lighting-inconsistent image

## Detection Methods Explained

### 1. Copy-Move Forgery Detection
- **Method**: Block-based correlation analysis
- **Detection**: Finds duplicated regions using template matching
- **Indicators**: High correlation between distant image blocks
- **Visualization**: Red and green rectangles around duplicated regions

### 2. Noise Pattern Analysis
- **Method**: High-pass filtering and variance analysis
- **Detection**: Statistical outliers in noise distribution
- **Indicators**: Regions with significantly different noise characteristics
- **Visualization**: Yellow rectangles around suspicious regions

### 3. JPEG Compression Artifacts
- **Method**: DCT (Discrete Cosine Transform) analysis
- **Detection**: Inconsistent compression patterns in 8x8 blocks
- **Indicators**: Unusual high-frequency components
- **Visualization**: Orange rectangles around suspicious blocks

### 4. Lighting Consistency Analysis
- **Method**: LAB color space brightness analysis
- **Detection**: Statistical analysis of lighting distribution
- **Indicators**: Regions with inconsistent brightness patterns
- **Visualization**: Purple rectangles around inconsistent regions

### 5. Edge Artifact Detection
- **Method**: Canny edge detection and contour analysis
- **Detection**: Suspicious edge patterns from splicing
- **Indicators**: Unnatural edge characteristics and shapes
- **Visualization**: Highlighted suspicious contours

## Interpretation of Results

### Confidence Scores
- **0.0 - 0.3**: Low likelihood of tampering
- **0.3 - 0.7**: Medium likelihood of tampering
- **0.7 - 1.0**: High likelihood of tampering

### Overall Assessment
- **Likely Tampered**: Boolean indicating if image is probably manipulated
- **Severity**: Low/Medium/High based on overall confidence
- **Individual Method Scores**: Detailed breakdown by detection method

## Output Files

### Visualizations
- `*_analysis.png`: Visual analysis showing detected regions
- Color-coded overlays for different detection methods
- Summary panel with overall assessment

### JSON Reports
- `tampering_analysis_results.json`: Comprehensive analysis data
- Detailed results for each detection method
- Numerical confidence scores and detected regions
- Image metadata and analysis parameters

## Technical Details

### Algorithms Used
- **Template Matching**: For copy-move detection
- **Statistical Analysis**: For noise and lighting inconsistencies
- **DCT Analysis**: For JPEG compression artifacts
- **Edge Detection**: Canny algorithm for edge artifacts
- **Color Space Conversion**: RGB to LAB for lighting analysis

### Performance Considerations
- **Block Size**: Optimized for different detection methods
- **Threshold Values**: Tuned for balance between sensitivity and false positives
- **Memory Usage**: Efficient processing for large images
- **Processing Time**: Varies based on image size and complexity

## Limitations

1. **False Positives**: Natural image features may trigger detection
2. **Sophisticated Tampering**: Advanced manipulation techniques may evade detection
3. **Image Quality**: Low-quality images may produce unreliable results
4. **Processing Time**: Large images require more processing time
5. **Format Dependencies**: Some methods work better with specific image formats

## Future Enhancements

- Deep learning-based detection models
- Real-time video tampering detection
- Advanced metadata analysis
- Integration with blockchain for authenticity verification
- Mobile app implementation

## Contributing
Feel free to contribute improvements, bug fixes, or additional detection methods. Please ensure all changes are well-documented and tested.

## License
This project is for educational and research purposes. Please ensure compliance with applicable laws and regulations when using for commercial purposes.
