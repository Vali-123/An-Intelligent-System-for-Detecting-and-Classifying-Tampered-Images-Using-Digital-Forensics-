#!/usr/bin/env python3
"""
📂 FOLDER SCAN - Batch Image Tampering Detection
Scan entire folders with progress tracking and summary report
"""

import os
import sys
import time
from pathlib import Path
from analyze_single_image import SingleImageTamperingDetector

class FolderScanner:
    def __init__(self):
        self.detector = SingleImageTamperingDetector()
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.results_summary = []
    
    def find_images(self, folder_path):
        """Find all supported images in folder"""
        images = []
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            return images
        
        for file_path in folder.rglob("*"):
            if file_path.suffix.lower() in self.supported_formats:
                images.append(file_path)
        
        return sorted(images)
    
    def scan_folder(self, folder_path):
        """Scan all images in folder with progress tracking"""
        print(f"📂 Scanning folder: {folder_path}")
        print("="*60)
        
        images = self.find_images(folder_path)
        
        if not images:
            print("❌ No supported images found!")
            print("Supported formats: JPG, PNG, BMP, TIFF")
            return
        
        print(f"🔍 Found {len(images)} images to analyze...")
        print("-" * 60)
        
        # Scan each image
        tampered_count = 0
        suspicious_count = 0
        clean_count = 0
        
        for i, image_path in enumerate(images, 1):
            # Progress indicator
            progress = i / len(images)
            bar_length = 30
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"\n📸 [{i:2d}/{len(images)}] {image_path.name}")
            print(f"Progress: [{bar}] {progress:.1%}")
            
            # Analyze image
            try:
                results = self.detector.analyze_image(str(image_path))
                confidence = results['overall_assessment']['tampering_confidence']
                likely_tampered = results['overall_assessment']['likely_tampered']
                
                # Categorize result
                if likely_tampered and confidence >= 0.7:
                    status = "🚨 TAMPERED"
                    tampered_count += 1
                elif confidence >= 0.3:
                    status = "⚠️ SUSPICIOUS"
                    suspicious_count += 1
                else:
                    status = "✅ CLEAN"
                    clean_count += 1
                
                print(f"Result: {status} (Confidence: {confidence:.1%})")
                
                # Store for summary
                self.results_summary.append({
                    'file': image_path.name,
                    'path': str(image_path),
                    'confidence': confidence,
                    'status': status,
                    'likely_tampered': likely_tampered
                })
                
            except Exception as e:
                print(f"❌ Error analyzing {image_path.name}: {str(e)}")
                continue
        
        # Show summary
        self.show_summary(tampered_count, suspicious_count, clean_count, len(images))
    
    def show_summary(self, tampered, suspicious, clean, total):
        """Display scan summary"""
        print("\n" + "="*60)
        print("📊 SCAN SUMMARY REPORT")
        print("="*60)
        
        # Visual summary
        print(f"\n📈 Results Breakdown:")
        print(f"  🚨 Tampered:   {tampered:2d} files ({tampered/total:.1%})")
        print(f"  ⚠️ Suspicious: {suspicious:2d} files ({suspicious/total:.1%})")
        print(f"  ✅ Clean:      {clean:2d} files ({clean/total:.1%})")
        print(f"  📊 Total:      {total:2d} files")
        
        # Risk assessment
        if tampered > 0:
            risk_level = "🔴 HIGH RISK"
            recommendation = "Immediate review recommended!"
        elif suspicious > total * 0.3:
            risk_level = "🟡 MEDIUM RISK"
            recommendation = "Some files need closer inspection."
        else:
            risk_level = "🟢 LOW RISK"
            recommendation = "Most files appear authentic."
        
        print(f"\n🎯 Overall Assessment: {risk_level}")
        print(f"💡 Recommendation: {recommendation}")
        
        # Top suspicious files
        if self.results_summary:
            suspicious_files = [r for r in self.results_summary if r['confidence'] >= 0.5]
            if suspicious_files:
                print(f"\n🔍 Top Suspicious Files:")
                sorted_files = sorted(suspicious_files, key=lambda x: x['confidence'], reverse=True)
                for i, file_info in enumerate(sorted_files[:5], 1):
                    print(f"  {i}. {file_info['file']} - {file_info['confidence']:.1%}")
        
        print("\n" + "="*60)
        print("🎉 Folder scan complete!")
        print("="*60)
    
    def save_report(self, folder_path):
        """Save detailed report to file"""
        report_file = Path(folder_path) / "tampering_scan_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("FOLDER TAMPERING SCAN REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Scan Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Folder: {folder_path}\n")
            f.write(f"Total Files: {len(self.results_summary)}\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 50 + "\n")
            
            for result in self.results_summary:
                f.write(f"File: {result['file']}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Confidence: {result['confidence']:.3f}\n")
                f.write(f"Path: {result['path']}\n")
                f.write("-" * 30 + "\n")
        
        print(f"📄 Detailed report saved: {report_file}")

def main():
    print("📂 FOLDER SCAN - Batch Image Tampering Detection")
    print("Scan entire folders for tampered images!")
    print("-" * 60)
    
    scanner = FolderScanner()
    
    if len(sys.argv) > 1:
        # Command line usage
        folder_path = sys.argv[1]
        scanner.scan_folder(folder_path)
        
        # Ask if user wants to save report
        save_report = input("\n💾 Save detailed report? (y/n): ").lower().strip()
        if save_report in ['y', 'yes']:
            scanner.save_report(folder_path)
    else:
        # Interactive mode
        print("\n📁 Enter folder path to scan:")
        folder_path = input("➤ ").strip().strip('"').strip("'")
        
        if folder_path and os.path.exists(folder_path):
            scanner.scan_folder(folder_path)
            
            # Ask if user wants to save report
            save_report = input("\n💾 Save detailed report? (y/n): ").lower().strip()
            if save_report in ['y', 'yes']:
                scanner.save_report(folder_path)
        else:
            print("❌ Invalid folder path. Try again!")

if __name__ == "__main__":
    main()
