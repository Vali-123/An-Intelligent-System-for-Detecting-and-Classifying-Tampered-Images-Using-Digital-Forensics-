import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
import json
from scipy import ndimage
from sklearn.cluster import KMeans
import sys
import warnings
warnings.filterwarnings('ignore')

class SingleImageTamperingDetector:
    def __init__(self):
        self.results = {}
        
    def load_image(self, image_path):
        """Load image using multiple methods for robustness"""
        try:
            # Try with OpenCV
            img_cv = cv2.imread(image_path)
            if img_cv is not None:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Try with PIL
            img_pil = Image.open(image_path)
            img_pil_array = np.array(img_pil.convert('RGB'))
            
            return img_cv if img_cv is not None else img_pil_array
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def detect_copy_move_forgery(self, image):
        """Detect copy-move forgery using block matching"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        block_size = 16
        threshold = 0.95
        
        matches = []
        blocks = {}
        
        # Extract overlapping blocks
        for i in range(0, height - block_size, 4):
            for j in range(0, width - block_size, 4):
                block = gray[i:i+block_size, j:j+block_size]
                block_hash = hash(block.tobytes())
                
                if block_hash in blocks:
                    # Calculate correlation
                    existing_block, (ex_i, ex_j) = blocks[block_hash]
                    correlation = cv2.matchTemplate(block, existing_block, cv2.TM_CCOEFF_NORMED)[0,0]
                    
                    if correlation > threshold and abs(i - ex_i) > block_size and abs(j - ex_j) > block_size:
                        matches.append(((i, j), (ex_i, ex_j), correlation))
                else:
                    blocks[block_hash] = (block, (i, j))
        
        confidence = min(len(matches) * 0.1, 1.0) if matches else 0.0
        return matches, confidence
    
    def analyze_noise_patterns(self, image):
        """Analyze noise distribution for tampering detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply high-pass filter to extract noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise = cv2.filter2D(gray, -1, kernel)
        
        # Divide image into blocks and analyze noise variance
        block_size = 32
        height, width = gray.shape
        variances = []
        positions = []
        
        for i in range(0, height - block_size, block_size):
            for j in range(0, width - block_size, block_size):
                block_noise = noise[i:i+block_size, j:j+block_size]
                var = np.var(block_noise)
                variances.append(var)
                positions.append((i, j))
        
        if len(variances) > 0:
            variances = np.array(variances)
            mean_var = np.mean(variances)
            std_var = np.std(variances)
            
            # Find outliers (potential tampered regions)
            outliers = []
            for i, var in enumerate(variances):
                if abs(var - mean_var) > 2 * std_var:
                    outliers.append((positions[i], var))
            
            confidence = min(len(outliers) * 0.05, 1.0) if outliers else 0.0
            return outliers, confidence
        
        return [], 0.0
    
    def detect_jpeg_compression_artifacts(self, image):
        """Detect inconsistent JPEG compression artifacts"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply DCT to 8x8 blocks (JPEG compression units)
        height, width = gray.shape
        artifacts = []
        
        for i in range(0, height - 8, 8):
            for j in range(0, width - 8, 8):
                block = gray[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # Analyze high-frequency components
                high_freq = np.sum(np.abs(dct_block[4:, 4:]))
                artifacts.append((i, j, high_freq))
        
        if artifacts:
            artifact_values = [a[2] for a in artifacts]
            mean_artifact = np.mean(artifact_values)
            std_artifact = np.std(artifact_values)
            
            suspicious_blocks = []
            for i, j, val in artifacts:
                if abs(val - mean_artifact) > 2 * std_artifact:
                    suspicious_blocks.append((i, j, val))
            
            confidence = min(len(suspicious_blocks) * 0.03, 1.0)
            return suspicious_blocks, confidence
        
        return [], 0.0
    
    def analyze_lighting_consistency(self, image):
        """Analyze lighting inconsistencies"""
        if len(image.shape) != 3:
            return [], 0.0
        
        # Convert to LAB color space for better lighting analysis
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Divide image into regions and analyze brightness
        height, width = l_channel.shape
        region_size = 50
        regions = []
        
        for i in range(0, height - region_size, region_size):
            for j in range(0, width - region_size, region_size):
                region = l_channel[i:i+region_size, j:j+region_size]
                mean_brightness = np.mean(region)
                std_brightness = np.std(region)
                regions.append((i, j, mean_brightness, std_brightness))
        
        if regions:
            brightnesses = [r[2] for r in regions]
            overall_mean = np.mean(brightnesses)
            overall_std = np.std(brightnesses)
            
            inconsistent_regions = []
            for i, j, brightness, std_bright in regions:
                if abs(brightness - overall_mean) > 2 * overall_std:
                    inconsistent_regions.append((i, j, brightness))
            
            confidence = min(len(inconsistent_regions) * 0.04, 1.0)
            return inconsistent_regions, confidence
        
        return [], 0.0
    
    def detect_edge_artifacts(self, image):
        """Detect edge artifacts that might indicate splicing"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        suspicious_edges = []
        for contour in contours:
            # Analyze contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 100 and perimeter > 50:
                # Calculate circularity (suspicious if too perfect)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Calculate moments for further analysis
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if circularity > 0.8 or area > 5000:  # Suspicious thresholds
                        suspicious_edges.append((cx, cy, area, circularity))
        
        confidence = min(len(suspicious_edges) * 0.1, 1.0)
        return suspicious_edges, confidence
    
    def analyze_image(self, image_path):
        """Main analysis function for single image"""
        print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
        
        image = self.load_image(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        results = {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "image_shape": list(image.shape),
            "analysis": {}
        }
        
        # Run all detection methods
        print("🔍 Detecting copy-move forgery...")
        copy_move_matches, cm_confidence = self.detect_copy_move_forgery(image)
        results["analysis"]["copy_move"] = {
            "matches": len(copy_move_matches),
            "confidence": float(cm_confidence),
            "description": "Detects duplicated regions within the image"
        }
        
        print("🔊 Analyzing noise patterns...")
        noise_outliers, noise_confidence = self.analyze_noise_patterns(image)
        results["analysis"]["noise_analysis"] = {
            "outliers": len(noise_outliers),
            "confidence": float(noise_confidence),
            "description": "Identifies inconsistent noise distributions"
        }
        
        print("📸 Detecting JPEG artifacts...")
        jpeg_artifacts, jpeg_confidence = self.detect_jpeg_compression_artifacts(image)
        results["analysis"]["jpeg_artifacts"] = {
            "suspicious_blocks": len(jpeg_artifacts),
            "confidence": float(jpeg_confidence),
            "description": "Analyzes compression inconsistencies"
        }
        
        print("💡 Analyzing lighting consistency...")
        lighting_issues, lighting_confidence = self.analyze_lighting_consistency(image)
        results["analysis"]["lighting"] = {
            "inconsistent_regions": len(lighting_issues),
            "confidence": float(lighting_confidence),
            "description": "Detects unnatural lighting variations"
        }
        
        print("🔍 Detecting edge artifacts...")
        edge_artifacts, edge_confidence = self.detect_edge_artifacts(image)
        results["analysis"]["edge_artifacts"] = {
            "suspicious_edges": len(edge_artifacts),
            "confidence": float(edge_confidence),
            "description": "Identifies suspicious edge patterns from splicing"
        }
        
        # Calculate overall tampering confidence
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
        
        return results

def main():
    """Main function to handle single image analysis"""
    detector = SingleImageTamperingDetector()
    
    # Check if image path provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Prompt for image path
        image_path = input("📁 Please enter the path to the image file: ").strip().strip('"')
    
    # Validate image path
    if not os.path.isfile(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        print("Please make sure the file exists and the path is correct.")
        return
    
    # Validate image format
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    if not any(image_path.lower().endswith(fmt) for fmt in supported_formats):
        print(f"❌ Error: Unsupported image format!")
        print(f"Supported formats: {', '.join(supported_formats)}")
        return
    
    print("\n" + "="*60)
    print("🔍 AI-BASED IMAGE TAMPERING DETECTION")
    print("="*60)
    
    # Run analysis
    results = detector.analyze_image(image_path)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Display results
    print(f"\n📊 ANALYSIS RESULTS")
    print("-"*30)
    print(f"📁 File: {results['image_name']}")
    print(f"📐 Size: {results['image_shape'][1]}x{results['image_shape'][0]} pixels")
    print(f"📈 Overall Confidence: {results['overall_assessment']['tampering_confidence']}")
    print(f"⚠️  Likely Tampered: {results['overall_assessment']['likely_tampered']}")
    print(f"🔴 Severity: {results['overall_assessment']['severity']}")
    
    print(f"\n💭 INTERPRETATION:")
    print(results['interpretation'])
    
    print(f"\n🔬 DETAILED BREAKDOWN:")
    print("-"*30)
    for method, data in results['analysis'].items():
        method_name = method.replace('_', ' ').title()
        print(f"{method_name}:")
        print(f"  • Confidence: {data['confidence']:.3f}")
        print(f"  • Description: {data['description']}")
        
        # Show specific findings
        if 'matches' in data and data['matches'] > 0:
            print(f"  • Found {data['matches']} suspicious matches")
        if 'outliers' in data and data['outliers'] > 0:
            print(f"  • Detected {data['outliers']} noise outliers")
        if 'suspicious_blocks' in data and data['suspicious_blocks'] > 0:
            print(f"  • Found {data['suspicious_blocks']} suspicious compression blocks")
        if 'inconsistent_regions' in data and data['inconsistent_regions'] > 0:
            print(f"  • Identified {data['inconsistent_regions']} inconsistent lighting regions")
        if 'suspicious_edges' in data and data['suspicious_edges'] > 0:
            print(f"  • Detected {data['suspicious_edges']} suspicious edge patterns")
    
    # Save results to JSON
    output_file = f"{os.path.splitext(results['image_name'])[0]}_tampering_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    print("\n" + "="*60)
    print("✅ Analysis completed!")
    print("="*60)

if __name__ == "__main__":
    main()
