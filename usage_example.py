#!/usr/bin/env python3
"""
Usage Example: Single Image Analysis
===================================

This script demonstrates how to use the image tampering detection system
to analyze a single image and get results in JSON format.
"""

from analyze_single_image import SingleImageTamperingDetector
import json
import os

def analyze_sample_images():
    """Analyze all sample images and show results"""
    detector = SingleImageTamperingDetector()
    
    # Sample images to analyze
    sample_images = [
        "authentic_image.jpg",
        "copy_move_tampered.jpg", 
        "spliced_tampered.jpg",
        "noise_tampered.jpg",
        "lighting_tampered.jpg"
    ]
    
    print("🔍 AI-BASED IMAGE TAMPERING DETECTION")
    print("="*60)
    print("📋 Analyzing sample images...")
    print()
    
    for image_name in sample_images:
        if not os.path.exists(image_name):
            print(f"⚠️  Skipping {image_name} - file not found")
            continue
            
        print(f"🔍 Analyzing: {image_name}")
        print("-" * 40)
        
        try:
            # Analyze the image
            results = detector.analyze_image(image_name)
            
            if "error" in results:
                print(f"❌ Error: {results['error']}")
                continue
            
            # Display key results
            confidence = results['overall_assessment']['tampering_confidence']
            likely_tampered = results['overall_assessment']['likely_tampered']
            severity = results['overall_assessment']['severity']
            
            # Status emoji
            status_emoji = "🚨" if severity == "High" else "⚠️" if severity == "Medium" else "✅"
            
            print(f"{status_emoji} Overall Confidence: {confidence:.3f}")
            print(f"⚠️  Likely Tampered: {'YES' if likely_tampered else 'NO'}")
            print(f"🔴 Severity: {severity}")
            
            # Show detection method results
            print("🔬 Detection Methods:")
            for method, data in results['analysis'].items():
                method_name = method.replace('_', ' ').title()
                print(f"   • {method_name}: {data['confidence']:.3f}")
            
            print(f"💭 {results['interpretation']}")
            
            # Save to individual JSON file
            output_file = f"{os.path.splitext(image_name)[0]}_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error analyzing {image_name}: {e}")
        
        print()  # Empty line between results

def demonstrate_json_output():
    """Show the JSON output format"""
    print("\n📋 JSON OUTPUT FORMAT EXAMPLE:")
    print("="*50)
    
    # Show example JSON structure
    example_json = {
        "image_path": "example_image.jpg",
        "image_name": "example_image.jpg", 
        "image_shape": [600, 800, 3],
        "analysis": {
            "copy_move": {
                "matches": 125,
                "confidence": 0.85,
                "description": "Detects duplicated regions within the image"
            },
            "noise_analysis": {
                "outliers": 15,
                "confidence": 0.45,
                "description": "Identifies inconsistent noise distributions"
            },
            "jpeg_artifacts": {
                "suspicious_blocks": 8,
                "confidence": 0.32,
                "description": "Analyzes compression inconsistencies"
            },
            "lighting": {
                "inconsistent_regions": 2,
                "confidence": 0.18,
                "description": "Detects unnatural lighting variations"
            },
            "edge_artifacts": {
                "suspicious_edges": 3,
                "confidence": 0.25,
                "description": "Identifies suspicious edge patterns from splicing"
            }
        },
        "overall_assessment": {
            "tampering_confidence": 0.610,
            "likely_tampered": True,
            "severity": "Medium"
        },
        "interpretation": "⚠️ MEDIUM likelihood of tampering detected. Some suspicious patterns found, requires closer inspection."
    }
    
    print(json.dumps(example_json, indent=2))

def show_usage_instructions():
    """Show how to use the system"""
    print("\n📋 USAGE INSTRUCTIONS:")
    print("="*50)
    print("1. 📁 Direct command line usage:")
    print('   python analyze_single_image.py "your_image.jpg"')
    print()
    print("2. 📁 Interactive demo:")
    print("   python demo.py")
    print()
    print("3. 🔧 Programmatic usage:")
    print("   from analyze_single_image import SingleImageTamperingDetector")
    print("   detector = SingleImageTamperingDetector()")
    print('   results = detector.analyze_image("image.jpg")')
    print()
    print("4. 📊 Supported formats:")
    print("   JPG, JPEG, PNG, BMP, TIFF, TIF")
    print()
    print("5. 📈 Confidence interpretation:")
    print("   • 0.0-0.3: Low likelihood of tampering")
    print("   • 0.3-0.7: Medium likelihood of tampering") 
    print("   • 0.7-1.0: High likelihood of tampering")

if __name__ == "__main__":
    print("🎯 IMAGE TAMPERING DETECTION - USAGE EXAMPLE")
    print("="*60)
    
    # Analyze sample images
    analyze_sample_images()
    
    # Show JSON format
    demonstrate_json_output()
    
    # Show usage instructions
    show_usage_instructions()
    
    print("\n" + "="*60)
    print("✅ Example completed! The system is ready to analyze your images.")
    print("="*60)
