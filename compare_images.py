#!/usr/bin/env python3
"""
🔄 IMAGE COMPARE - Side-by-Side Tampering Detection
Compare two images and see which one might be tampered
"""

import os
import sys
from analyze_single_image import SingleImageTamperingDetector

class ImageComparer:
    def __init__(self):
        self.detector = SingleImageTamperingDetector()
    
    def analyze_both(self, image1_path, image2_path):
        """Analyze both images and compare results"""
        
        # Check if files exist
        if not os.path.exists(image1_path):
            print(f"❌ Cannot find: {image1_path}")
            return
        
        if not os.path.exists(image2_path):
            print(f"❌ Cannot find: {image2_path}")
            return
        
        print("🔄 COMPARING TWO IMAGES")
        print("="*60)
        
        # Analyze first image
        print(f"\n📸 Analyzing Image 1: {os.path.basename(image1_path)}")
        print("-" * 40)
        results1 = self.detector.analyze_image(image1_path)
        conf1 = results1['overall_assessment']['tampering_confidence']
        tampered1 = results1['overall_assessment']['likely_tampered']
        
        # Analyze second image  
        print(f"\n📸 Analyzing Image 2: {os.path.basename(image2_path)}")
        print("-" * 40)
        results2 = self.detector.analyze_image(image2_path)
        conf2 = results2['overall_assessment']['tampering_confidence']
        tampered2 = results2['overall_assessment']['likely_tampered']
        
        # Side-by-side comparison
        self.show_comparison(image1_path, image2_path, conf1, conf2, tampered1, tampered2, results1, results2)
    
    def show_comparison(self, path1, path2, conf1, conf2, tampered1, tampered2, results1, results2):
        """Display side-by-side comparison"""
        
        print("\n" + "🔄 SIDE-BY-SIDE COMPARISON".center(60))
        print("="*60)
        
        name1 = os.path.basename(path1)[:25]
        name2 = os.path.basename(path2)[:25]
        
        print(f"│ {'IMAGE 1':<28} │ {'IMAGE 2':<28} │")
        print(f"│ {name1:<28} │ {name2:<28} │")
        print("├─────────────────────────────┼─────────────────────────────┤")
        
        # Confidence comparison
        print(f"│ Confidence: {conf1:>6.1%}           │ Confidence: {conf2:>6.1%}           │")
        
        # Status comparison
        status1 = "🚨 TAMPERED" if tampered1 else "✅ CLEAN"
        status2 = "🚨 TAMPERED" if tampered2 else "✅ CLEAN" 
        print(f"│ Status: {status1:<17} │ Status: {status2:<17} │")
        
        # Confidence bars
        bar1 = self.make_confidence_bar(conf1)
        bar2 = self.make_confidence_bar(conf2)
        print(f"│ {bar1:<28} │ {bar2:<28} │")
        
        print("└─────────────────────────────┴─────────────────────────────┘")
        
        # Winner determination
        self.declare_winner(conf1, conf2, name1, name2)
        
        # Method breakdown
        self.show_method_breakdown(results1, results2, name1, name2)
    
    def make_confidence_bar(self, confidence):
        """Create visual confidence bar"""
        bar_length = 15
        filled = int(bar_length * confidence)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        if confidence < 0.3:
            color = "🟢"
        elif confidence < 0.7:
            color = "🟡"  
        else:
            color = "🔴"
        
        return f"{color}[{bar}]{confidence:.0%}"
    
    def declare_winner(self, conf1, conf2, name1, name2):
        """Determine which image is more suspicious"""
        print(f"\n🏆 COMPARISON RESULT:")
        
        if abs(conf1 - conf2) < 0.1:
            print("🤝 Both images have similar tampering likelihood!")
            print("   The difference is too small to determine a clear winner.")
        elif conf1 > conf2:
            diff = conf1 - conf2
            print(f"⚠️  Image 1 ({name1}) appears MORE suspicious!")
            print(f"   Confidence difference: +{diff:.1%}")
            if conf1 > 0.7:
                print("   🚨 High likelihood of tampering detected!")
        else:
            diff = conf2 - conf1
            print(f"⚠️  Image 2 ({name2}) appears MORE suspicious!")
            print(f"   Confidence difference: +{diff:.1%}")
            if conf2 > 0.7:
                print("   🚨 High likelihood of tampering detected!")
    
    def show_method_breakdown(self, results1, results2, name1, name2):
        """Show detailed method comparison"""
        print(f"\n📊 DETECTION METHOD BREAKDOWN:")
        print("-" * 60)
        
        methods = [
            ('copy_move', 'Copy-Move Detection'),
            ('noise_analysis', 'Noise Analysis'),
            ('jpeg_artifacts', 'JPEG Artifacts'),
            ('lighting', 'Lighting Analysis'),
            ('edge_artifacts', 'Edge Artifacts')
        ]
        
        print(f"{'Method':<20} │ {name1[:12]:<12} │ {name2[:12]:<12} │ Winner")
        print("─" * 60)
        
        for method_key, method_name in methods:
            conf1 = results1['analysis'][method_key]['confidence']
            conf2 = results2['analysis'][method_key]['confidence']
            
            if conf1 > conf2 + 0.1:
                winner = f"📸 {name1[:8]}"
            elif conf2 > conf1 + 0.1:
                winner = f"📸 {name2[:8]}"
            else:
                winner = "🤝 Tie"
            
            print(f"{method_name[:19]:<20} │ {conf1:>7.1%}      │ {conf2:>7.1%}      │ {winner}")

def main():
    print("🔄 IMAGE COMPARE - Side-by-Side Tampering Detection")
    print("Compare two images to see which might be tampered!")
    print("-" * 60)
    
    comparer = ImageComparer()
    
    if len(sys.argv) >= 3:
        # Command line usage
        image1_path = sys.argv[1]
        image2_path = sys.argv[2]
        comparer.analyze_both(image1_path, image2_path)
    else:
        # Interactive mode
        print("\n📸 Enter path to first image:")
        image1_path = input("Image 1 ➤ ").strip().strip('"').strip("'")
        
        if not image1_path or not os.path.exists(image1_path):
            print("❌ Invalid first image path!")
            return
        
        print("\n📸 Enter path to second image:")
        image2_path = input("Image 2 ➤ ").strip().strip('"').strip("'")
        
        if not image2_path or not os.path.exists(image2_path):
            print("❌ Invalid second image path!")
            return
        
        comparer.analyze_both(image1_path, image2_path)

if __name__ == "__main__":
    main()
