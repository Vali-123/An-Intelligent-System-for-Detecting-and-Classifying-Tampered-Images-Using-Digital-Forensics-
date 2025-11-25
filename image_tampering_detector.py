import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
import json
from scipy import ndimage
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ImageTamperingDetector:
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
        """Main analysis function"""
        print(f"Analyzing image: {image_path}")
        
        image = self.load_image(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        results = {
            "image_path": image_path,
            "image_shape": image.shape,
            "analysis": {}
        }
        
        # Run all detection methods
        print("Detecting copy-move forgery...")
        copy_move_matches, cm_confidence = self.detect_copy_move_forgery(image)
        results["analysis"]["copy_move"] = {
            "matches": len(copy_move_matches),
            "confidence": cm_confidence,
            "details": copy_move_matches[:5]  # Limit output
        }
        
        print("Analyzing noise patterns...")
        noise_outliers, noise_confidence = self.analyze_noise_patterns(image)
        results["analysis"]["noise_analysis"] = {
            "outliers": len(noise_outliers),
            "confidence": noise_confidence,
            "details": noise_outliers[:5]
        }
        
        print("Detecting JPEG artifacts...")
        jpeg_artifacts, jpeg_confidence = self.detect_jpeg_compression_artifacts(image)
        results["analysis"]["jpeg_artifacts"] = {
            "suspicious_blocks": len(jpeg_artifacts),
            "confidence": jpeg_confidence,
            "details": jpeg_artifacts[:5]
        }
        
        print("Analyzing lighting consistency...")
        lighting_issues, lighting_confidence = self.analyze_lighting_consistency(image)
        results["analysis"]["lighting"] = {
            "inconsistent_regions": len(lighting_issues),
            "confidence": lighting_confidence,
            "details": lighting_issues[:5]
        }
        
        print("Detecting edge artifacts...")
        edge_artifacts, edge_confidence = self.detect_edge_artifacts(image)
        results["analysis"]["edge_artifacts"] = {
            "suspicious_edges": len(edge_artifacts),
            "confidence": edge_confidence,
            "details": edge_artifacts[:5]
        }
        
        # Calculate overall tampering confidence
        confidences = [cm_confidence, noise_confidence, jpeg_confidence, 
                      lighting_confidence, edge_confidence]
        overall_confidence = np.mean(confidences)
        
        results["overall_assessment"] = {
            "tampering_confidence": overall_confidence,
            "likely_tampered": overall_confidence > 0.3,
            "severity": "High" if overall_confidence > 0.7 else "Medium" if overall_confidence > 0.3 else "Low"
        }
        
        return results
    
    def visualize_results(self, image_path, results):
        """Create visualization of detected tampering"""
        image = self.load_image(image_path)
        if image is None:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Image Tampering Analysis: {os.path.basename(image_path)}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Copy-move detection
        copy_move_img = image.copy()
        if results["analysis"]["copy_move"]["matches"] > 0:
            for match in results["analysis"]["copy_move"]["details"]:
                if len(match) >= 2:
                    (i1, j1), (i2, j2) = match[0], match[1]
                    cv2.rectangle(copy_move_img, (j1, i1), (j1+16, i1+16), (255, 0, 0), 2)
                    cv2.rectangle(copy_move_img, (j2, i2), (j2+16, i2+16), (0, 255, 0), 2)
        
        axes[0, 1].imshow(copy_move_img)
        axes[0, 1].set_title(f'Copy-Move Detection\nConfidence: {results["analysis"]["copy_move"]["confidence"]:.2f}')
        axes[0, 1].axis('off')
        
        # Noise analysis
        noise_img = image.copy()
        if results["analysis"]["noise_analysis"]["outliers"] > 0:
            for outlier in results["analysis"]["noise_analysis"]["details"]:
                if len(outlier) >= 1:
                    (i, j) = outlier[0]
                    cv2.rectangle(noise_img, (j, i), (j+32, i+32), (255, 255, 0), 2)
        
        axes[0, 2].imshow(noise_img)
        axes[0, 2].set_title(f'Noise Analysis\nConfidence: {results["analysis"]["noise_analysis"]["confidence"]:.2f}')
        axes[0, 2].axis('off')
        
        # JPEG artifacts
        jpeg_img = image.copy()
        if results["analysis"]["jpeg_artifacts"]["suspicious_blocks"] > 0:
            for block in results["analysis"]["jpeg_artifacts"]["details"]:
                if len(block) >= 2:
                    i, j = block[0], block[1]
                    cv2.rectangle(jpeg_img, (j, i), (j+8, i+8), (255, 165, 0), 2)
        
        axes[1, 0].imshow(jpeg_img)
        axes[1, 0].set_title(f'JPEG Artifacts\nConfidence: {results["analysis"]["jpeg_artifacts"]["confidence"]:.2f}')
        axes[1, 0].axis('off')
        
        # Lighting analysis
        lighting_img = image.copy()
        if results["analysis"]["lighting"]["inconsistent_regions"] > 0:
            for region in results["analysis"]["lighting"]["details"]:
                if len(region) >= 2:
                    i, j = region[0], region[1]
                    cv2.rectangle(lighting_img, (j, i), (j+50, i+50), (128, 0, 128), 2)
        
        axes[1, 1].imshow(lighting_img)
        axes[1, 1].set_title(f'Lighting Analysis\nConfidence: {results["analysis"]["lighting"]["confidence"]:.2f}')
        axes[1, 1].axis('off')
        
        # Overall assessment
        axes[1, 2].text(0.1, 0.8, f'Overall Confidence: {results["overall_assessment"]["tampering_confidence"]:.2f}', 
                       fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.6, f'Likely Tampered: {results["overall_assessment"]["likely_tampered"]}', 
                       fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.4, f'Severity: {results["overall_assessment"]["severity"]}', 
                       fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Overall Assessment')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        output_path = image_path.replace('.', '_analysis.')
        if not output_path.endswith('.png'):
            output_path = output_path.rsplit('.', 1)[0] + '_analysis.png'
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualization saved as: {output_path}")

def main():
    detector = ImageTamperingDetector()
    
    # Look for images in current directory
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    current_dir = os.getcwd()
    
    image_files = []
    for file in os.listdir(current_dir):
        if any(file.lower().endswith(fmt) for fmt in supported_formats):
            image_files.append(file)
    
    if not image_files:
        print("No image files found in current directory.")
        print("Please add some images to analyze.")
        return
    
    print(f"Found {len(image_files)} image file(s):")
    for i, file in enumerate(image_files):
        print(f"{i+1}. {file}")
    
    # Analyze all images
    all_results = []
    for image_file in image_files:
        print(f"\n{'='*50}")
        results = detector.analyze_image(image_file)
        all_results.append(results)
        
        # Print summary
        if "error" not in results:
            print(f"\nSUMMARY FOR {image_file}:")
            print(f"Overall Tampering Confidence: {results['overall_assessment']['tampering_confidence']:.2f}")
            print(f"Likely Tampered: {results['overall_assessment']['likely_tampered']}")
            print(f"Severity: {results['overall_assessment']['severity']}")
            
            # Create visualization
            try:
                detector.visualize_results(image_file, results)
            except Exception as e:
                print(f"Could not create visualization: {e}")
        else:
            print(f"Error analyzing {image_file}: {results['error']}")
    
    # Save detailed results
    with open('tampering_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: tampering_analysis_results.json")

if __name__ == "__main__":
    main()
