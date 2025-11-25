import os
import glob
import json
from quality_based_detector import QualityBasedTamperingDetector
import time

def batch_test_images(folder_path, output_file=None):
    """Test multiple images in a folder"""
    detector = QualityBasedTamperingDetector()
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.gif', '*.webp']
    
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to analyze...")
    print("=" * 60)
    
    results = []
    summary_stats = {
        'total_images': len(image_files),
        'high_risk': 0,
        'medium_risk': 0,
        'low_risk': 0,
        'processing_time': 0
    }
    
    start_time = time.time()
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Analyzing: {os.path.basename(image_path)}")
        
        try:
            # Analyze image
            result = detector.analyze_image_quality(image_path)
            
            if "error" not in result:
                # Extract key information
                tampering_prob = result['tampering_assessment']['tampering_probability']
                verdict = result['tampering_assessment']['verdict']
                quality_score = result['quality_metrics']['overall_quality_score']
                
                # Categorize risk
                if tampering_prob > 0.7:
                    risk_category = "HIGH"
                    summary_stats['high_risk'] += 1
                elif tampering_prob > 0.4:
                    risk_category = "MEDIUM"
                    summary_stats['medium_risk'] += 1
                else:
                    risk_category = "LOW"
                    summary_stats['low_risk'] += 1
                
                print(f"   Result: {verdict}")
                print(f"   Risk: {risk_category} ({tampering_prob:.1%})")
                print(f"   Quality: {quality_score:.3f}/1.000")
                
                # Store result
                results.append({
                    'filename': os.path.basename(image_path),
                    'full_path': image_path,
                    'verdict': verdict,
                    'tampering_probability': tampering_prob,
                    'quality_score': quality_score,
                    'risk_category': risk_category,
                    'full_analysis': result
                })
            else:
                print(f"   Error: {result['error']}")
                results.append({
                    'filename': os.path.basename(image_path),
                    'full_path': image_path,
                    'error': result['error']
                })
        
        except Exception as e:
            print(f"   Exception: {str(e)}")
            results.append({
                'filename': os.path.basename(image_path),
                'full_path': image_path,
                'exception': str(e)
            })
    
    end_time = time.time()
    summary_stats['processing_time'] = end_time - start_time
    
    # Print summary
    print(f"\n" + "=" * 60)
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Images Processed: {summary_stats['total_images']}")
    print(f"Processing Time: {summary_stats['processing_time']:.1f} seconds")
    print(f"Average Time per Image: {summary_stats['processing_time']/summary_stats['total_images']:.1f} seconds")
    print(f"\nRisk Distribution:")
    print(f"  🔴 High Risk: {summary_stats['high_risk']} ({summary_stats['high_risk']/summary_stats['total_images']*100:.1f}%)")
    print(f"  🟡 Medium Risk: {summary_stats['medium_risk']} ({summary_stats['medium_risk']/summary_stats['total_images']*100:.1f}%)")
    print(f"  🟢 Low Risk: {summary_stats['low_risk']} ({summary_stats['low_risk']/summary_stats['total_images']*100:.1f}%)")
    
    # Show top high-risk images
    high_risk_images = [r for r in results if r.get('risk_category') == 'HIGH']
    if high_risk_images:
        print(f"\n🚨 HIGH RISK IMAGES:")
        for img in sorted(high_risk_images, key=lambda x: x.get('tampering_probability', 0), reverse=True)[:5]:
            print(f"   • {img['filename']} - {img.get('tampering_probability', 0):.1%} risk")
    
    # Show top quality images
    quality_images = [r for r in results if 'quality_score' in r]
    if quality_images:
        print(f"\n✨ HIGHEST QUALITY IMAGES:")
        for img in sorted(quality_images, key=lambda x: x.get('quality_score', 0), reverse=True)[:5]:
            print(f"   • {img['filename']} - {img.get('quality_score', 0):.3f} quality")
    
    # Save results if requested
    if output_file:
        full_results = {
            'summary': summary_stats,
            'individual_results': results,
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'folder_analyzed': folder_path
        }
        
        with open(output_file, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {output_file}")
    
    return results, summary_stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch test multiple images for quality-based tampering detection')
    parser.add_argument('folder', help='Folder containing images to analyze')
    parser.add_argument('--output', '-o', help='Output JSON file for detailed results')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode - minimal output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' not found!")
        return
    
    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a directory!")
        return
    
    print("🔍 Quality-Based Image Tampering Detection - Batch Mode")
    print(f"📁 Analyzing folder: {args.folder}")
    
    results, stats = batch_test_images(args.folder, args.output)
    
    if args.quick:
        print(f"\nQuick Summary: {stats['high_risk']} high-risk, {stats['medium_risk']} medium-risk, {stats['low_risk']} low-risk images")

if __name__ == "__main__":
    main()
