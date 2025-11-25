import os
import sys
import argparse
import json
from ml_tampering_detector import MLTamperingDetector

def main():
    parser = argparse.ArgumentParser(description='Test a single image for tampering detection')
    parser.add_argument('image_path', help='Path to the image file to analyze')
    parser.add_argument('--output', '-o', help='Output file to save results (JSON format)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found!")
        sys.exit(1)
    
    # Initialize detector
    print("Initializing ML Tampering Detector...")
    detector = MLTamperingDetector()
    
    # Load models
    if not detector.load_models():
        print("Error: Could not load trained models!")
        print("Please run 'python ml_tampering_detector.py' to train the models first.")
        sys.exit(1)
    
    print(f"Analyzing image: {args.image_path}")
    print("-" * 50)
    
    # Analyze the image
    try:
        result = detector.predict_image(args.image_path)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        
        # Display results
        print_results(result, args.verbose)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

def print_results(result, verbose=False):
    """Print analysis results in a formatted way"""
    print(f"Image: {os.path.basename(result['image_path'])}")
    print("=" * 50)
    
    # Overall assessment
    recommendation = result['recommendation']
    ensemble = result['predictions']['ensemble']
    
    print(f"FINAL VERDICT: {recommendation}")
    print(f"Confidence: {ensemble['confidence']:.3f}")
    print(f"Tampering Probability: {ensemble['tampered_probability']:.3f}")
    
    # Risk level
    tampered_prob = ensemble['tampered_probability']
    if tampered_prob > 0.8:
        risk_level = "VERY HIGH 🔴"
    elif tampered_prob > 0.6:
        risk_level = "HIGH 🟠"
    elif tampered_prob > 0.4:
        risk_level = "MEDIUM 🟡"
    else:
        risk_level = "LOW 🟢"
    
    print(f"Risk Level: {risk_level}")
    
    if verbose:
        print(f"\nDETAILED PREDICTIONS:")
        print("-" * 30)
        
        for model_name, pred_data in result['predictions'].items():
            model_display = model_name.replace('_', ' ').title()
            print(f"\n{model_display}:")
            print(f"  Prediction: {pred_data['prediction']}")
            print(f"  Confidence: {pred_data['confidence']:.3f}")
            print(f"  Tampered Probability: {pred_data['tampered_probability']:.3f}")
    
    print(f"\nRECOMMENDATION:")
    if recommendation == "Tampered":
        print("⚠️  WARNING: This image shows signs of tampering!")
        print("   → Verify authenticity before use")
        print("   → Consider additional verification methods")
        print("   → Use caution if using for important purposes")
    else:
        print("✅ This image appears to be authentic")
        print("   → Safe to use based on current analysis")
        print("   → Consider context and source verification")

def test_sample_images():
    """Test with sample images from the dataset"""
    detector = MLTamperingDetector()
    
    if not detector.load_models():
        print("Error: Could not load trained models!")
        return
    
    print("Testing Sample Images from Dataset")
    print("=" * 50)
    
    # Test original image
    original_path = os.path.join("celebrity_dataset", "original", "celebrity_01_Leonardo_DiCaprio.jpg")
    if os.path.exists(original_path):
        print(f"\n🔍 Testing ORIGINAL image:")
        result = detector.predict_image(original_path)
        print_results(result)
    
    # Test tampered image
    tampered_path = os.path.join("celebrity_dataset", "tampered", "tampered_celebrity_01_Leonardo_DiCaprio.jpg")
    if os.path.exists(tampered_path):
        print(f"\n🔍 Testing TAMPERED image:")
        result = detector.predict_image(tampered_path)
        print_results(result)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run sample tests
        test_sample_images()
    else:
        main()
