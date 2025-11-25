#!/usr/bin/env python3
"""
🚀 QUICK SCAN - Fast Image Tampering Detection
Simple drag-and-drop style detection with fun visual feedback
"""

import os
import sys
import time
import random
from analyze_single_image import SingleImageTamperingDetector

class QuickScanner:
    def __init__(self):
        self.detector = SingleImageTamperingDetector()
        self.scanning_emojis = ["🔍", "🕵️", "🔎", "👁️", "🤖", "⚡", "🎯"]
        self.result_emojis = {
            "authentic": ["✅", "😇", "👍", "🟢", "💚"],
            "suspicious": ["⚠️", "🤔", "🟡", "🔍", "👀"],
            "tampered": ["🚨", "😱", "❌", "🔴", "💀"]
        }
    
    def animate_scanning(self, duration=3):
        """Fun scanning animation"""
        print("\n" + "="*50)
        print("🚀 QUICK SCAN IN PROGRESS...")
        print("="*50)
        
        for i in range(duration * 4):
            emoji = random.choice(self.scanning_emojis)
            dots = "." * ((i % 3) + 1)
            print(f"\r{emoji} Analyzing{dots}   ", end="", flush=True)
            time.sleep(0.25)
        
        print("\r🎉 Analysis Complete!     ")
    
    def get_fun_verdict(self, confidence, likely_tampered):
        """Generate fun verdict messages"""
        if not likely_tampered:
            emoji = random.choice(self.result_emojis["authentic"])
            messages = [
                "This image looks legit! 📸",
                "Clean as a whistle! 🎵",
                "Authentic vibes detected! ✨",
                "No funny business here! 😊",
                "Original content confirmed! 🌟"
            ]
        elif confidence < 0.5:
            emoji = random.choice(self.result_emojis["suspicious"])
            messages = [
                "Hmm, something's a bit fishy... 🐟",
                "My spider senses are tingling! 🕷️",
                "Looks suspicious, but not sure... 🤷",
                "Might want to double-check this one! 🔍",
                "Gray area detected! 🌫️"
            ]
        else:
            emoji = random.choice(self.result_emojis["tampered"])
            messages = [
                "BUSTED! Someone's been busy! 🚔",
                "Tampering detected! Red alert! 🚨",
                "This image has been doctored! 💊",
                "Fake news alert! 📰",
                "Photo manipulation confirmed! 🎭"
            ]
        
        return f"{emoji} {random.choice(messages)}"
    
    def quick_scan(self, image_path):
        """Perform quick scan with fun interface"""
        if not os.path.exists(image_path):
            print("❌ Oops! Can't find that image file!")
            return
        
        # Fun scanning animation
        self.animate_scanning()
        
        # Actual analysis
        print("\n🔬 Running AI detection algorithms...")
        results = self.detector.analyze_image(image_path)
        
        # Fun results display
        confidence = results['overall_assessment']['tampering_confidence']
        likely_tampered = results['overall_assessment']['likely_tampered']
        
        print("\n" + "🎯 QUICK SCAN RESULTS 🎯".center(50))
        print("="*50)
        
        # Fun verdict
        verdict = self.get_fun_verdict(confidence, likely_tampered)
        print(f"\n{verdict}")
        
        # Simple confidence bar
        print(f"\n📊 Confidence: {confidence:.1%}")
        bar_length = 20
        filled = int(bar_length * confidence)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        if confidence < 0.3:
            color = "🟢"
        elif confidence < 0.7:
            color = "🟡"
        else:
            color = "🔴"
        
        print(f"{color} [{bar}] {confidence:.1%}")
        
        # Top detection methods
        print(f"\n🔍 Top Detections:")
        analysis = results['analysis']
        sorted_methods = sorted(analysis.items(), key=lambda x: x[1]['confidence'], reverse=True)
        
        for i, (method, data) in enumerate(sorted_methods[:3]):
            method_name = method.replace('_', ' ').title()
            conf = data['confidence']
            if conf > 0.5:
                icon = "🚨"
            elif conf > 0.3:
                icon = "⚠️"
            else:
                icon = "✅"
            print(f"  {i+1}. {icon} {method_name}: {conf:.1%}")
        
        print("\n" + "="*50)
        print("🎉 Quick scan complete! Have a great day! 🌟")
        print("="*50)

def main():
    print("🚀 QUICK SCAN - Simple Image Tampering Detection")
    print("Just drag and drop, or type the path!")
    print("-" * 50)
    
    scanner = QuickScanner()
    
    if len(sys.argv) > 1:
        # Command line usage
        image_path = sys.argv[1]
        scanner.quick_scan(image_path)
    else:
        # Interactive mode
        print("\n📁 Enter image path (or drag & drop file here):")
        image_path = input("➤ ").strip().strip('"').strip("'")
        
        if image_path:
            scanner.quick_scan(image_path)
        else:
            print("❌ No image provided. Try again!")

if __name__ == "__main__":
    main()
