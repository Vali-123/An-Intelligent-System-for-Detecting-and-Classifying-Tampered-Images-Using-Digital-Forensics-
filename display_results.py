import json
import os
from datetime import datetime

def display_analysis_summary():
    """Display a comprehensive summary of the image tampering analysis results"""
    
    # Load the results
    with open('tampering_analysis_results.json', 'r') as f:
        results = json.load(f)
    
    print("="*80)
    print("🔍 AI-BASED IMAGE TAMPERING DETECTION RESULTS")
    print("="*80)
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total images analyzed: {len(results)}")
    print("="*80)
    
    # Summary statistics
    high_confidence = 0
    medium_confidence = 0
    low_confidence = 0
    likely_tampered = 0
    
    for result in results:
        confidence = result['overall_assessment']['tampering_confidence']
        if confidence > 0.7:
            high_confidence += 1
        elif confidence > 0.3:
            medium_confidence += 1
        else:
            low_confidence += 1
            
        if result['overall_assessment']['likely_tampered'] == 'True':
            likely_tampered += 1
    
    print("\n📊 OVERALL STATISTICS:")
    print("-"*40)
    print(f"Images likely tampered: {likely_tampered}/{len(results)}")
    print(f"High confidence detections: {high_confidence}")
    print(f"Medium confidence detections: {medium_confidence}")
    print(f"Low confidence detections: {low_confidence}")
    
    # Detailed results for each image
    for i, result in enumerate(results, 1):
        print(f"\n\n🖼️  IMAGE {i}: {result['image_path']}")
        print("="*60)
        
        # Overall assessment
        assessment = result['overall_assessment']
        confidence = assessment['tampering_confidence']
        severity = assessment['severity']
        tampered = assessment['likely_tampered']
        
        # Color coding for severity
        severity_emoji = "🔴" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
        tampered_emoji = "⚠️" if tampered == 'True' else "✅"
        
        print(f"{tampered_emoji} LIKELY TAMPERED: {tampered}")
        print(f"{severity_emoji} SEVERITY: {severity}")
        print(f"📈 OVERALL CONFIDENCE: {confidence:.3f}")
        print(f"📐 IMAGE SIZE: {result['image_shape'][1]}x{result['image_shape'][0]} pixels")
        
        # Detection method breakdown
        print("\n🔬 DETECTION METHOD BREAKDOWN:")
        print("-"*40)
        
        analysis = result['analysis']
        
        # Copy-move detection
        cm = analysis['copy_move']
        print(f"📋 Copy-Move Forgery:")
        print(f"   • Matches found: {cm['matches']}")
        print(f"   • Confidence: {cm['confidence']:.3f}")
        
        # Noise analysis
        noise = analysis['noise_analysis']
        print(f"🔊 Noise Pattern Analysis:")
        print(f"   • Outliers detected: {noise['outliers']}")
        print(f"   • Confidence: {noise['confidence']:.3f}")
        
        # JPEG artifacts
        jpeg = analysis['jpeg_artifacts']
        print(f"📸 JPEG Compression Artifacts:")
        print(f"   • Suspicious blocks: {jpeg['suspicious_blocks']}")
        print(f"   • Confidence: {jpeg['confidence']:.3f}")
        
        # Lighting analysis
        lighting = analysis['lighting']
        print(f"💡 Lighting Consistency:")
        print(f"   • Inconsistent regions: {lighting['inconsistent_regions']}")
        print(f"   • Confidence: {lighting['confidence']:.3f}")
        
        # Edge artifacts
        edges = analysis['edge_artifacts']
        print(f"🔍 Edge Artifacts:")
        print(f"   • Suspicious edges: {edges['suspicious_edges']}")
        print(f"   • Confidence: {edges['confidence']:.3f}")
        
        # Interpretation
        print(f"\n💭 INTERPRETATION:")
        print("-"*20)
        if confidence > 0.7:
            print("🚨 HIGH likelihood of tampering detected!")
            print("   Multiple detection methods show strong evidence of manipulation.")
        elif confidence > 0.3:
            print("⚠️  MEDIUM likelihood of tampering detected.")
            print("   Some suspicious patterns found, requires closer inspection.")
        else:
            print("✅ LOW likelihood of tampering.")
            print("   Image appears authentic or contains minimal suspicious patterns.")
        
        # Key findings
        key_findings = []
        if cm['matches'] > 100:
            key_findings.append("High number of copy-move matches detected")
        if noise['outliers'] > 20:
            key_findings.append("Significant noise pattern inconsistencies")
        if jpeg['suspicious_blocks'] > 50:
            key_findings.append("Many suspicious JPEG compression blocks")
        if lighting['inconsistent_regions'] > 5:
            key_findings.append("Notable lighting inconsistencies")
        if edges['suspicious_edges'] > 2:
            key_findings.append("Multiple suspicious edge patterns")
        
        if key_findings:
            print(f"\n🔑 KEY FINDINGS:")
            for finding in key_findings:
                print(f"   • {finding}")
    
    # File information
    print(f"\n\n📁 OUTPUT FILES GENERATED:")
    print("-"*40)
    print("• tampering_analysis_results.json - Detailed analysis data")
    
    # List visualization files
    for result in results:
        image_name = result['image_path'].replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        viz_file = f"{image_name}_analysis_analysis.png"
        if os.path.exists(viz_file):
            print(f"• {viz_file} - Visual analysis")
    
    print(f"\n📋 TECHNICAL NOTES:")
    print("-"*20)
    print("• Confidence scores range from 0.0 (no tampering) to 1.0 (definitely tampered)")
    print("• Multiple detection methods are combined for overall assessment")
    print("• Visual analysis files show color-coded suspicious regions")
    print("• Red/Green rectangles: Copy-move matches")
    print("• Yellow rectangles: Noise inconsistencies") 
    print("• Orange rectangles: JPEG artifacts")
    print("• Purple rectangles: Lighting inconsistencies")
    
    print("\n" + "="*80)
    print("Analysis complete! Check the visualization files for detailed visual results.")
    print("="*80)

if __name__ == "__main__":
    display_analysis_summary()
