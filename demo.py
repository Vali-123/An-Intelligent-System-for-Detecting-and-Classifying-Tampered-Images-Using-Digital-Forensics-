#!/usr/bin/env python3
"""
Interactive Image Tampering Detection Demo
==========================================

Simple interactive demo for testing the image tampering detection system.
Just run this script and follow the prompts to analyze any image.
"""

import os
import sys
from analyze_single_image import SingleImageTamperingDetector
import json

def print_banner():
    """Print the demo banner"""
    print("\n" + "="*70)
    print("🔍 AI-BASED IMAGE TAMPERING DETECTION DEMO")
    print("="*70)
    print("📋 Upload any image and get instant tampering analysis!")
    print("🎯 Supports: JPG, JPEG, PNG, BMP, TIFF formats")
    print("="*70)

def get_image_files_in_directory():
    """Get all image files in current directory"""
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    current_dir = os.getcwd()
    
    image_files = []
    for file in os.listdir(current_dir):
        if any(file.lower().endswith(fmt) for fmt in supported_formats):
            image_files.append(file)
    
    return sorted(image_files)

def select_from_existing_images():
    """Allow user to select from existing images in directory"""
    image_files = get_image_files_in_directory()
    
    if not image_files:
        print("❌ No image files found in current directory.")
        return None
    
    print(f"\n📁 Found {len(image_files)} image(s) in current directory:")
    print("-" * 50)
    
    for i, file in enumerate(image_files, 1):
        file_size = os.path.getsize(file)
        print(f"{i:2d}. {file:<30} ({file_size:,} bytes)")
    
    while True:
        try:
            choice = input(f"\n🔢 Select image (1-{len(image_files)}) or 0 to cancel: ").strip()
            if choice == '0':
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(image_files):
                return image_files[choice - 1]
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(image_files)} or 0.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def get_image_path():
    """Get image path from user - either by selection or manual entry"""
    print("\n📁 How would you like to select an image?")
    print("1. 📂 Choose from images in current directory")
    print("2. 📝 Enter image path manually")
    print("3. ❌ Exit demo")
    
    while True:
        choice = input("\n🔢 Enter your choice (1-3): ").strip()
        
        if choice == '1':
            return select_from_existing_images()
        
        elif choice == '2':
            image_path = input("\n📁 Enter the full path to your image: ").strip().strip('"')
            if os.path.isfile(image_path):
                return image_path
            else:
                print(f"❌ File not found: {image_path}")
                continue
        
        elif choice == '3':
            return 'EXIT'
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

def display_results_summary(results):
    """Display a beautiful summary of results"""
    print("\n" + "🎯" + "="*68 + "🎯")
    print("🔍 TAMPERING DETECTION RESULTS")
    print("🎯" + "="*68 + "🎯")
    
    # Basic info
    print(f"📁 Image: {results['image_name']}")
    print(f"📐 Dimensions: {results['image_shape'][1]}×{results['image_shape'][0]} pixels")
    print(f"📊 Channels: {results['image_shape'][2]}")
    
    # Overall assessment with emoji indicators
    confidence = results['overall_assessment']['tampering_confidence']
    likely_tampered = results['overall_assessment']['likely_tampered']
    severity = results['overall_assessment']['severity']
    
    # Status indicators
    status_emoji = "🚨" if severity == "High" else "⚠️" if severity == "Medium" else "✅"
    confidence_bar = "█" * int(confidence * 20)  # 20-character bar
    
    print(f"\n{status_emoji} OVERALL ASSESSMENT:")
    print("-" * 30)
    print(f"🎯 Tampering Confidence: {confidence:.3f}")
    print(f"📊 Confidence Bar: [{confidence_bar:<20}] {confidence:.1%}")
    print(f"⚠️ Likely Tampered: {'YES' if likely_tampered else 'NO'}")
    print(f"🔴 Severity Level: {severity}")
    
    print(f"\n💭 AI INTERPRETATION:")
    print(f"   {results['interpretation']}")
    
    # Method breakdown with visual indicators
    print(f"\n🔬 DETECTION METHOD BREAKDOWN:")
    print("=" * 50)
    
    methods = [
        ("Copy-Move Detection", "copy_move", "🔍"),
        ("Noise Analysis", "noise_analysis", "🔊"),
        ("JPEG Artifacts", "jpeg_artifacts", "📸"),
        ("Lighting Analysis", "lighting", "💡"),
        ("Edge Artifacts", "edge_artifacts", "🔍")
    ]
    
    for name, key, emoji in methods:
        data = results['analysis'][key]
        conf = data['confidence']
        conf_bar = "█" * int(conf * 10)  # 10-character bar
        
        print(f"{emoji} {name}:")
        print(f"   📊 Confidence: {conf:.3f} [{conf_bar:<10}] {conf:.1%}")
        print(f"   📝 {data['description']}")
        
        # Show specific findings
        findings = []
        if 'matches' in data and data['matches'] > 0:
            findings.append(f"{data['matches']} matches found")
        if 'outliers' in data and data['outliers'] > 0:
            findings.append(f"{data['outliers']} outliers detected")
        if 'suspicious_blocks' in data and data['suspicious_blocks'] > 0:
            findings.append(f"{data['suspicious_blocks']} suspicious blocks")
        if 'inconsistent_regions' in data and data['inconsistent_regions'] > 0:
            findings.append(f"{data['inconsistent_regions']} inconsistent regions")
        if 'suspicious_edges' in data and data['suspicious_edges'] > 0:
            findings.append(f"{data['suspicious_edges']} suspicious edges")
        
        if findings:
            print(f"   🔎 Findings: {', '.join(findings)}")
        else:
            print(f"   ✅ No suspicious patterns detected")
        print()

def main():
    """Main demo function"""
    print_banner()
    detector = SingleImageTamperingDetector()
    
    while True:
        image_path = get_image_path()
        
        if image_path == 'EXIT':
            print("\n👋 Thanks for using the Image Tampering Detection Demo!")
            break
        
        if image_path is None:
            continue
        
        # Validate image format
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        if not any(image_path.lower().endswith(fmt) for fmt in supported_formats):
            print(f"❌ Unsupported format! Supported: {', '.join(supported_formats)}")
            continue
        
        print(f"\n🚀 Analyzing image: {os.path.basename(image_path)}")
        print("⏳ Please wait while AI analyzes the image...")
        
        try:
            # Run analysis
            results = detector.analyze_image(image_path)
            
            if "error" in results:
                print(f"❌ Error: {results['error']}")
                continue
            
            # Display beautiful results
            display_results_summary(results)
            
            # Save results
            output_file = f"{os.path.splitext(os.path.basename(image_path))[0]}_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Detailed results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ An error occurred during analysis: {e}")
        
        # Ask if user wants to analyze another image
        print("\n" + "="*70)
        another = input("🔄 Would you like to analyze another image? (y/n): ").strip().lower()
        
        if another not in ['y', 'yes']:
            print("\n🎉 Demo completed! Thank you for using the AI Image Tampering Detector!")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("Please check your installation and try again.")
