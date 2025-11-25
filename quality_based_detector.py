import cv2
import numpy as np
import os
from PIL import Image, ImageFilter
import json
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class QualityBasedTamperingDetector:
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
    
    def calculate_blur_metric(self, image):
        """Calculate blur level using Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Calculate Laplacian variance (measure of blur)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize blur score (higher = less blur, lower = more blur)
        blur_score = min(laplacian_var / 500.0, 1.0)  # Normalize to 0-1
        
        return blur_score, laplacian_var
    
    def calculate_sharpness_metric(self, image):
        """Calculate image sharpness using gradient magnitude"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sharpness = np.mean(gradient_magnitude)
        
        # Normalize sharpness score
        sharpness_score = min(sharpness / 50.0, 1.0)  # Normalize to 0-1
        
        return sharpness_score, sharpness
    
    def calculate_noise_level(self, image):
        """Calculate noise level in the image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply noise extraction filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise = cv2.filter2D(gray, -1, kernel)
        
        # Calculate noise level
        noise_level = np.std(noise)
        
        # Normalize noise score (higher noise = more likely tampered)
        noise_score = min(noise_level / 30.0, 1.0)
        
        return noise_score, noise_level
    
    def calculate_compression_quality(self, image):
        """Estimate JPEG compression quality"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Analyze 8x8 DCT blocks for compression artifacts
        h, w = gray.shape
        block_variances = []
        
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = gray[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # Calculate high-frequency energy (compression artifacts)
                high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))
                block_variances.append(high_freq_energy)
        
        if block_variances:
            compression_artifacts = np.std(block_variances)
            # Normalize compression score (higher artifacts = lower quality)
            compression_score = max(0, 1 - (compression_artifacts / 100.0))
        else:
            compression_score = 0.5
        
        return compression_score, compression_artifacts if block_variances else 0
    
    def calculate_resolution_quality(self, image):
        """Calculate resolution-based quality metrics"""
        height, width = image.shape[:2]
        total_pixels = height * width
        
        # Define quality thresholds
        if total_pixels >= 2073600:  # 1920x1080 or higher
            resolution_score = 1.0
            resolution_category = "High Resolution (HD+)"
        elif total_pixels >= 921600:  # 1280x720
            resolution_score = 0.8
            resolution_category = "Medium Resolution (HD)"
        elif total_pixels >= 307200:  # 640x480
            resolution_score = 0.6
            resolution_category = "Standard Resolution"
        else:
            resolution_score = 0.3
            resolution_category = "Low Resolution"
        
        return resolution_score, resolution_category, (width, height)
    
    def calculate_color_quality(self, image):
        """Calculate color distribution quality"""
        if len(image.shape) != 3:
            return 0.5, "Grayscale"
        
        # Calculate color variance across channels
        color_vars = []
        for channel in range(3):
            var = np.var(image[:, :, channel])
            color_vars.append(var)
        
        # Good color distribution should have reasonable variance
        avg_color_var = np.mean(color_vars)
        color_score = min(avg_color_var / 2000.0, 1.0)
        
        # Check for color consistency
        color_consistency = 1.0 - (np.std(color_vars) / max(np.mean(color_vars), 1))
        
        return color_score, color_consistency
    
    def analyze_image_quality(self, image_path):
        """Main analysis function focusing on image quality"""
        print(f"Analyzing image quality: {image_path}")
        
        image = self.load_image(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Calculate various quality metrics
        blur_score, blur_value = self.calculate_blur_metric(image)
        sharpness_score, sharpness_value = self.calculate_sharpness_metric(image)
        noise_score, noise_value = self.calculate_noise_level(image)
        compression_score, compression_artifacts = self.calculate_compression_quality(image)
        resolution_score, resolution_category, dimensions = self.calculate_resolution_quality(image)
        color_score, color_consistency = self.calculate_color_quality(image)
        
        # Calculate overall quality score
        quality_weights = {
            'blur': 0.25,
            'sharpness': 0.25, 
            'noise': 0.15,
            'compression': 0.15,
            'resolution': 0.10,
            'color': 0.10
        }
        
        overall_quality = (
            blur_score * quality_weights['blur'] +
            sharpness_score * quality_weights['sharpness'] +
            (1 - noise_score) * quality_weights['noise'] +  # Lower noise = higher quality
            compression_score * quality_weights['compression'] +
            resolution_score * quality_weights['resolution'] +
            color_score * quality_weights['color']
        )
        
        # Determine tampering likelihood based on quality
        # High quality images are less likely to be tampered
        # Low quality images might indicate tampering or poor source
        
        if overall_quality >= 0.8:
            tampering_probability = max(0.05, 0.3 - overall_quality)  # Very low chance
            tampering_verdict = "Not Tampered (High Quality)"
            risk_level = "Very Low"
            confidence = 0.95
        elif overall_quality >= 0.6:
            tampering_probability = 0.3
            tampering_verdict = "Likely Not Tampered (Good Quality)"
            risk_level = "Low"
            confidence = 0.80
        elif overall_quality >= 0.4:
            tampering_probability = 0.5
            tampering_verdict = "Uncertain (Medium Quality)"
            risk_level = "Medium"
            confidence = 0.60
        elif overall_quality >= 0.2:
            tampering_probability = 0.7
            tampering_verdict = "Possibly Tampered (Poor Quality)"
            risk_level = "High"
            confidence = 0.75
        else:
            tampering_probability = 0.9
            tampering_verdict = "Likely Tampered (Very Poor Quality)"
            risk_level = "Very High"
            confidence = 0.85
        
        # Special case for very blurry images
        if blur_score < 0.2:
            tampering_probability = max(tampering_probability, 0.8)
            tampering_verdict = "Likely Tampered (Excessive Blur)"
            risk_level = "Very High"
        
        results = {
            "image_path": image_path,
            "image_dimensions": dimensions,
            "quality_metrics": {
                "overall_quality_score": round(overall_quality, 3),
                "blur_score": round(blur_score, 3),
                "blur_value": round(blur_value, 2),
                "sharpness_score": round(sharpness_score, 3),
                "sharpness_value": round(sharpness_value, 2),
                "noise_score": round(noise_score, 3),
                "noise_value": round(noise_value, 2),
                "compression_score": round(compression_score, 3),
                "compression_artifacts": round(compression_artifacts, 2),
                "resolution_score": round(resolution_score, 3),
                "resolution_category": resolution_category,
                "color_score": round(color_score, 3),
                "color_consistency": round(color_consistency, 3)
            },
            "tampering_assessment": {
                "verdict": tampering_verdict,
                "tampering_probability": round(tampering_probability, 3),
                "confidence": round(confidence, 3),
                "risk_level": risk_level
            },
            "quality_analysis": {
                "is_blurry": blur_score < 0.3,
                "is_noisy": noise_score > 0.6,
                "is_low_resolution": resolution_score < 0.5,
                "has_compression_artifacts": compression_score < 0.4,
                "overall_assessment": self.get_quality_assessment(overall_quality)
            }
        }
        
        return results
    
    def get_quality_assessment(self, quality_score):
        """Get human-readable quality assessment"""
        if quality_score >= 0.8:
            return "Excellent - High quality image with minimal artifacts"
        elif quality_score >= 0.6:
            return "Good - Acceptable quality with minor issues"
        elif quality_score >= 0.4:
            return "Fair - Moderate quality with noticeable issues"
        elif quality_score >= 0.2:
            return "Poor - Low quality with significant artifacts"
        else:
            return "Very Poor - Extremely low quality, likely corrupted or heavily processed"
    
    def print_detailed_results(self, results):
        """Print formatted results"""
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"IMAGE QUALITY-BASED TAMPERING ANALYSIS")
        print(f"{'='*60}")
        print(f"Image: {os.path.basename(results['image_path'])}")
        print(f"Dimensions: {results['image_dimensions'][0]}x{results['image_dimensions'][1]}")
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        print(f"   Verdict: {results['tampering_assessment']['verdict']}")
        print(f"   Tampering Probability: {results['tampering_assessment']['tampering_probability']:.1%}")
        print(f"   Confidence: {results['tampering_assessment']['confidence']:.1%}")
        print(f"   Risk Level: {results['tampering_assessment']['risk_level']}")
        
        print(f"\n📊 QUALITY METRICS:")
        metrics = results['quality_metrics']
        print(f"   Overall Quality Score: {metrics['overall_quality_score']:.3f}/1.000")
        print(f"   Blur Score: {metrics['blur_score']:.3f} (Blur Value: {metrics['blur_value']:.1f})")
        print(f"   Sharpness Score: {metrics['sharpness_score']:.3f} (Sharpness: {metrics['sharpness_value']:.1f})")
        print(f"   Noise Score: {metrics['noise_score']:.3f} (Noise Level: {metrics['noise_value']:.1f})")
        print(f"   Compression Score: {metrics['compression_score']:.3f}")
        print(f"   Resolution: {metrics['resolution_category']}")
        print(f"   Color Quality: {metrics['color_score']:.3f}")
        
        print(f"\n🔍 QUALITY ANALYSIS:")
        analysis = results['quality_analysis']
        print(f"   Overall Assessment: {analysis['overall_assessment']}")
        print(f"   Is Blurry: {'Yes' if analysis['is_blurry'] else 'No'}")
        print(f"   Is Noisy: {'Yes' if analysis['is_noisy'] else 'No'}")
        print(f"   Is Low Resolution: {'Yes' if analysis['is_low_resolution'] else 'No'}")
        print(f"   Has Compression Artifacts: {'Yes' if analysis['has_compression_artifacts'] else 'No'}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if results['tampering_assessment']['tampering_probability'] > 0.7:
            print(f"   ⚠️  HIGH RISK: This image shows significant quality issues!")
            print(f"   → Image may be tampered, corrupted, or from unreliable source")
            print(f"   → Verify authenticity before use")
            print(f"   → Consider finding higher quality version")
        elif results['tampering_assessment']['tampering_probability'] > 0.4:
            print(f"   ⚡ MEDIUM RISK: Image quality is questionable")
            print(f"   → Use with caution for important purposes")
            print(f"   → Consider source reliability")
        else:
            print(f"   ✅ LOW RISK: Image appears to be of good quality")
            print(f"   → Likely authentic based on quality metrics")
            print(f"   → Safe to use for most purposes")

def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Quality-based image tampering detection')
    parser.add_argument('image_path', nargs='?', help='Path to the image file to analyze')
    parser.add_argument('--output', '-o', help='Output file to save results (JSON format)')
    
    args = parser.parse_args()
    
    detector = QualityBasedTamperingDetector()
    
    if args.image_path:
        # Analyze specific image
        if not os.path.exists(args.image_path):
            print(f"Error: Image file '{args.image_path}' not found!")
            sys.exit(1)
        
        results = detector.analyze_image_quality(args.image_path)
        detector.print_detailed_results(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    else:
        # Test with sample images if available
        print("Quality-Based Tampering Detection System")
        print("=" * 50)
        print("Usage: python quality_based_detector.py <image_path>")
        print("\nExample:")
        print("python quality_based_detector.py my_image.jpg")
        print("python quality_based_detector.py my_image.jpg --output analysis.json")

if __name__ == "__main__":
    main()
