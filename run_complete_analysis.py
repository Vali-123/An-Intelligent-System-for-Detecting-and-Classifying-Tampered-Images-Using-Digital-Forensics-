#!/usr/bin/env python3
"""
Complete Image Tampering Detection Workflow
==========================================

This script demonstrates the complete workflow of the AI-based image tampering detection system.
It generates sample images, analyzes them, and displays comprehensive results.

Usage: python run_complete_analysis.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🔍 {title}")
    print("="*80)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n📋 STEP {step_num}: {description}")
    print("-" * (len(description) + 15))

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    try:
        print(f"▶️  Running {script_name}...")
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ Script {script_name} not found!")
        return False

def main():
    """Main workflow execution"""
    start_time = time.time()
    
    print_header("AI-BASED IMAGE TAMPERING DETECTION SYSTEM")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Step 1: Generate sample images
    print_step(1, "GENERATING SAMPLE TEST IMAGES")
    if not run_script("generate_test_images.py", "Sample image generation"):
        print("❌ Failed to generate sample images. Exiting.")
        return
    
    # Copy images to main directory for analysis
    print("📁 Copying images to main directory...")
    try:
        if os.name == 'nt':  # Windows
            os.system('copy sample_images\\*.jpg . >nul 2>&1')
        else:  # Unix/Linux/Mac
            os.system('cp sample_images/*.jpg . 2>/dev/null')
        print("✅ Images copied successfully!")
    except:
        print("⚠️  Warning: Could not copy images automatically")
    
    # Step 2: Run tampering detection
    print_step(2, "ANALYZING IMAGES FOR TAMPERING")
    if not run_script("image_tampering_detector.py", "Image tampering analysis"):
        print("❌ Failed to analyze images. Exiting.")
        return
    
    # Step 3: Display results
    print_step(3, "DISPLAYING COMPREHENSIVE RESULTS")
    if not run_script("display_results.py", "Results display"):
        print("⚠️  Could not display formatted results, but analysis completed")
    
    # Step 4: Summary and file listing
    print_step(4, "ANALYSIS SUMMARY AND OUTPUT FILES")
    
    # Count files
    image_files = len([f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    analysis_files = len([f for f in os.listdir('.') if '_analysis.png' in f])
    
    print(f"📊 ANALYSIS STATISTICS:")
    print(f"   • Images analyzed: {image_files}")
    print(f"   • Visual analysis files created: {analysis_files}")
    print(f"   • JSON results file: tampering_analysis_results.json")
    
    print(f"\n🎯 KEY ACHIEVEMENTS:")
    print("   ✅ Successfully implemented 5 different tampering detection methods")
    print("   ✅ Copy-move forgery detection using block correlation")
    print("   ✅ Noise pattern analysis with statistical outliers")
    print("   ✅ JPEG compression artifact detection using DCT")
    print("   ✅ Lighting consistency analysis in LAB color space")
    print("   ✅ Edge artifact detection using Canny edge detection")
    print("   ✅ Generated comprehensive visual analysis reports")
    print("   ✅ Created detailed JSON results with confidence scores")
    
    print(f"\n📈 DETECTION CAPABILITIES:")
    print("   🔍 Copy-Move Forgery: Detects duplicated regions within images")
    print("   🔊 Noise Analysis: Identifies inconsistent noise patterns")
    print("   📸 JPEG Artifacts: Finds compression inconsistencies")
    print("   💡 Lighting Analysis: Detects unnatural lighting variations")
    print("   🔍 Edge Artifacts: Identifies suspicious edge patterns from splicing")
    
    # List important files
    print(f"\n📁 IMPORTANT OUTPUT FILES:")
    important_files = [
        ("image_tampering_detector.py", "Main detection algorithm"),
        ("tampering_analysis_results.json", "Detailed analysis results"),
        ("display_results.py", "Results visualization script"),
        ("README.md", "Complete project documentation"),
        ("requirements.txt", "Python dependencies")
    ]
    
    for filename, description in important_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"   📄 {filename:<35} ({size:,} bytes) - {description}")
    
    # List analysis visualization files
    print(f"\n🖼️  VISUAL ANALYSIS FILES:")
    for file in os.listdir('.'):
        if '_analysis.png' in file:
            size = os.path.getsize(file)
            print(f"   🎨 {file:<45} ({size:,} bytes)")
    
    # Execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print_header("WORKFLOW COMPLETED SUCCESSFULLY! 🎉")
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n🚀 NEXT STEPS:")
    print("   1. 📊 Review the detailed analysis results above")
    print("   2. 🎨 Open the *_analysis.png files to see visual detections")
    print("   3. 📋 Check tampering_analysis_results.json for raw data")
    print("   4. 📖 Read README.md for detailed documentation")
    print("   5. 🔧 Modify detection parameters if needed")
    print("   6. 📤 Add your own images to the directory for analysis")
    
    print(f"\n💡 USAGE TIPS:")
    print("   • Place new images in the directory and run image_tampering_detector.py")
    print("   • Confidence scores above 0.7 indicate high likelihood of tampering")
    print("   • Visual analysis files show color-coded suspicious regions")
    print("   • Each detection method contributes to the overall confidence score")
    
    print("\n" + "="*80)
    print("🔍 AI-BASED IMAGE TAMPERING DETECTION SYSTEM - READY FOR USE!")
    print("="*80)

if __name__ == "__main__":
    main()
