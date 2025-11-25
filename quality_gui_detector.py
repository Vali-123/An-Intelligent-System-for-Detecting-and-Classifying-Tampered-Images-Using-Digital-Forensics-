import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
from quality_based_detector import QualityBasedTamperingDetector
import threading

class QualityGUIDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Quality-Based Image Tampering Detection")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize the detector
        self.detector = QualityBasedTamperingDetector()
        
        # Variables
        self.current_image_path = None
        self.current_result = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main title
        title_label = tk.Label(
            self.root, 
            text="Quality-Based Image Tampering Detection System", 
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=15)
        
        # Subtitle
        subtitle_label = tk.Label(
            self.root, 
            text="Detect tampering based on image quality metrics (blur, noise, compression, etc.)", 
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        subtitle_label.pack(pady=(0, 15))
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel for controls and image
        left_panel = tk.Frame(main_frame, bg='#f0f0f0')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Control buttons frame
        control_frame = tk.Frame(left_panel, bg='#ecf0f1', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(control_frame, text="Controls", font=("Arial", 14, "bold"), 
                bg='#ecf0f1', fg='#2c3e50').pack(pady=10)
        
        # Button frame for horizontal layout
        button_frame = tk.Frame(control_frame, bg='#ecf0f1')
        button_frame.pack(pady=10)
        
        self.load_btn = tk.Button(
            button_frame, 
            text="📁 Load Image", 
            command=self.load_image,
            font=("Arial", 12),
            bg='#3498db',
            fg='white',
            width=15,
            height=2
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = tk.Button(
            button_frame, 
            text="🔍 Analyze Quality", 
            command=self.analyze_image,
            font=("Arial", 12),
            bg='#e74c3c',
            fg='white',
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(
            button_frame, 
            text="💾 Save Results", 
            command=self.save_results,
            font=("Arial", 12),
            bg='#27ae60',
            fg='white',
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Status and progress
        status_frame = tk.Frame(control_frame, bg='#ecf0f1')
        status_frame.pack(pady=10)
        
        self.status_label = tk.Label(
            status_frame, 
            text="Ready to analyze images", 
            font=("Arial", 11),
            bg='#ecf0f1',
            fg='#7f8c8d'
        )
        self.status_label.pack()
        
        self.progress = ttk.Progressbar(
            status_frame, 
            mode='indeterminate',
            length=300
        )
        self.progress.pack(pady=5)
        
        # Image display frame
        image_frame = tk.Frame(left_panel, bg='#ffffff', relief=tk.RAISED, bd=2)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(image_frame, text="Image Preview", font=("Arial", 14, "bold"), 
                bg='#ffffff', fg='#2c3e50').pack(pady=10)
        
        self.image_label = tk.Label(
            image_frame, 
            text="No image loaded\\n\\nClick 'Load Image' to select an image\\nfrom your computer or downloaded from Google", 
            bg='#ffffff',
            fg='#95a5a6',
            font=("Arial", 14),
            justify=tk.CENTER
        )
        self.image_label.pack(expand=True)
        
        # Right panel for results
        right_panel = tk.Frame(main_frame, bg='#ffffff', relief=tk.RAISED, bd=2, width=500)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_panel.pack_propagate(False)
        
        tk.Label(right_panel, text="Analysis Results", font=("Arial", 14, "bold"), 
                bg='#ffffff', fg='#2c3e50').pack(pady=10)
        
        # Results notebook for tabbed interface
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary tab
        self.summary_frame = tk.Frame(self.notebook, bg='#ffffff')
        self.notebook.add(self.summary_frame, text="📊 Summary")
        
        self.summary_text = tk.Text(
            self.summary_frame, 
            font=("Consolas", 11),
            bg='#f8f9fa',
            fg='#2c3e50',
            wrap=tk.WORD
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Quality Metrics tab
        self.metrics_frame = tk.Frame(self.notebook, bg='#ffffff')
        self.notebook.add(self.metrics_frame, text="🔬 Quality Metrics")
        
        self.metrics_text = tk.Text(
            self.metrics_frame, 
            font=("Consolas", 10),
            bg='#f8f9fa',
            fg='#2c3e50',
            wrap=tk.WORD
        )
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Recommendations tab
        self.recommendations_frame = tk.Frame(self.notebook, bg='#ffffff')
        self.notebook.add(self.recommendations_frame, text="💡 Recommendations")
        
        self.recommendations_text = tk.Text(
            self.recommendations_frame, 
            font=("Arial", 11),
            bg='#f8f9fa',
            fg='#2c3e50',
            wrap=tk.WORD
        )
        self.recommendations_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbars to text widgets
        for text_widget in [self.summary_text, self.metrics_text, self.recommendations_text]:
            scrollbar = tk.Scrollbar(text_widget.master)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=text_widget.yview)
    
    def load_image(self):
        """Load an image file"""
        file_types = [
            ("All Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif *.webp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select an image file (including images downloaded from Google)",
            filetypes=file_types
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.config(state=tk.NORMAL)
            self.update_status(f"Image loaded: {os.path.basename(file_path)}")
            
            # Clear previous results
            self.clear_results()
            self.save_btn.config(state=tk.DISABLED)
    
    def display_image(self, image_path):
        """Display the loaded image"""
        try:
            # Load and resize image for display
            pil_image = Image.open(image_path)
            
            # Get original dimensions for display
            orig_width, orig_height = pil_image.size
            
            # Resize to fit in the display area while maintaining aspect ratio
            max_width, max_height = 600, 400
            
            # Calculate scaling factor
            scale_w = max_width / orig_width
            scale_h = max_height / orig_height
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale
            
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update the label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {str(e)}")
    
    def analyze_image(self):
        """Analyze the loaded image for quality and tampering"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        self.update_status("Analyzing image quality...")
        self.progress.start()
        self.analyze_btn.config(state=tk.DISABLED)
        
        # Run analysis in a separate thread to avoid freezing UI
        thread = threading.Thread(target=self._perform_analysis)
        thread.daemon = True
        thread.start()
    
    def _perform_analysis(self):
        """Perform the actual analysis"""
        try:
            # Analyze the image
            result = self.detector.analyze_image_quality(self.current_image_path)
            self.current_result = result
            
            # Update UI in main thread
            self.root.after(0, self._analysis_complete, result)
            
        except Exception as e:
            self.root.after(0, self._analysis_error, str(e))
    
    def _analysis_complete(self, result):
        """Handle analysis completion"""
        self.progress.stop()
        self.analyze_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        
        if "error" in result:
            self.update_status(f"Analysis failed: {result['error']}")
            messagebox.showerror("Error", f"Analysis failed: {result['error']}")
            return
        
        # Display results
        self.display_results(result)
        self.update_status("Analysis complete!")
    
    def _analysis_error(self, error_msg):
        """Handle analysis error"""
        self.progress.stop()
        self.analyze_btn.config(state=tk.NORMAL)
        self.update_status(f"Analysis failed: {error_msg}")
        messagebox.showerror("Error", f"Analysis failed: {error_msg}")
    
    def display_results(self, result):
        """Display analysis results in the tabs"""
        # Clear all text widgets
        self.clear_results()
        
        # Summary tab
        self.display_summary(result)
        
        # Quality Metrics tab
        self.display_metrics(result)
        
        # Recommendations tab
        self.display_recommendations(result)
    
    def display_summary(self, result):
        """Display summary results"""
        summary = f"""🎯 TAMPERING ANALYSIS SUMMARY
{'='*50}

Image: {os.path.basename(result['image_path'])}
Dimensions: {result['image_dimensions'][0]} × {result['image_dimensions'][1]} pixels

🏆 FINAL VERDICT
{result['tampering_assessment']['verdict']}

📊 KEY METRICS
Tampering Probability: {result['tampering_assessment']['tampering_probability']:.1%}
Analysis Confidence: {result['tampering_assessment']['confidence']:.1%}
Risk Level: {result['tampering_assessment']['risk_level']}
Overall Quality Score: {result['quality_metrics']['overall_quality_score']:.3f}/1.000

🔍 QUALITY ISSUES DETECTED
"""
        
        analysis = result['quality_analysis']
        issues = []
        if analysis['is_blurry']:
            issues.append("• Image is blurry")
        if analysis['is_noisy']:
            issues.append("• High noise levels detected")
        if analysis['is_low_resolution']:
            issues.append("• Low resolution image")
        if analysis['has_compression_artifacts']:
            issues.append("• Compression artifacts present")
        
        if issues:
            summary += "\n".join(issues)
        else:
            summary += "✅ No major quality issues detected"
        
        summary += f"\n\n📋 OVERALL ASSESSMENT\n{analysis['overall_assessment']}"
        
        self.summary_text.insert(tk.END, summary)
        
        # Color code the verdict
        self.highlight_verdict()
    
    def display_metrics(self, result):
        """Display detailed quality metrics"""
        metrics = result['quality_metrics']
        
        metrics_text = f"""🔬 DETAILED QUALITY METRICS
{'='*50}

📐 RESOLUTION ANALYSIS
Category: {metrics['resolution_category']}
Score: {metrics['resolution_score']:.3f}/1.000

🌫️ BLUR ANALYSIS
Blur Score: {metrics['blur_score']:.3f}/1.000
Laplacian Variance: {metrics['blur_value']:.1f}
Status: {"⚠️ BLURRY" if metrics['blur_score'] < 0.3 else "✅ SHARP"}

⚡ SHARPNESS ANALYSIS
Sharpness Score: {metrics['sharpness_score']:.3f}/1.000
Gradient Magnitude: {metrics['sharpness_value']:.1f}

🔊 NOISE ANALYSIS
Noise Score: {metrics['noise_score']:.3f}/1.000
Noise Level: {metrics['noise_value']:.1f}
Status: {"⚠️ NOISY" if metrics['noise_score'] > 0.6 else "✅ CLEAN"}

📷 COMPRESSION ANALYSIS
Compression Score: {metrics['compression_score']:.3f}/1.000
Artifacts Level: {metrics['compression_artifacts']:.1f}
Status: {"⚠️ COMPRESSED" if metrics['compression_score'] < 0.4 else "✅ GOOD QUALITY"}

🎨 COLOR ANALYSIS
Color Score: {metrics['color_score']:.3f}/1.000
Color Consistency: {metrics['color_consistency']:.3f}/1.000

📊 SCORING BREAKDOWN
• Blur contributes 25% to overall score
• Sharpness contributes 25% to overall score  
• Noise contributes 15% to overall score
• Compression contributes 15% to overall score
• Resolution contributes 10% to overall score
• Color quality contributes 10% to overall score
"""
        
        self.metrics_text.insert(tk.END, metrics_text)
    
    def display_recommendations(self, result):
        """Display recommendations and explanations"""
        tampering_prob = result['tampering_assessment']['tampering_probability']
        
        recommendations = f"""💡 RECOMMENDATIONS & EXPLANATIONS
{'='*50}

"""
        
        if tampering_prob > 0.7:
            recommendations += """🚨 HIGH RISK IMAGE - USE WITH EXTREME CAUTION

This image shows significant quality issues that may indicate:
• The image has been tampered or manipulated
• The image is from an unreliable or low-quality source
• The image has been heavily compressed or processed
• The image may be corrupted or damaged

⚠️ IMMEDIATE ACTIONS RECOMMENDED:
• Do not use this image for important purposes
• Verify the source and authenticity
• Try to obtain a higher quality version
• Consider alternative images
• If this is evidence, flag for manual review

🔍 TECHNICAL EXPLANATION:
Poor image quality often correlates with tampering because:
- Manipulation software often reduces image quality
- Multiple save/edit cycles degrade images
- Fake or synthesized content tends to have quality issues
- Legitimate high-quality images are less likely to be problematic
"""
        
        elif tampering_prob > 0.4:
            recommendations += """⚡ MEDIUM RISK IMAGE - USE WITH CAUTION

This image has moderate quality issues that suggest:
• Possible minor tampering or editing
• Average quality source or multiple processing steps
• Some compression or quality degradation
• May be legitimate but from lower quality source

⚠️ SUGGESTED ACTIONS:
• Use caution for important applications
• Consider the source reliability
• Look for higher quality alternatives if available
• Apply additional verification if needed

🔍 TECHNICAL EXPLANATION:
Medium quality images may indicate normal wear/processing
but could also suggest minor manipulation or poor source quality.
"""
        
        else:
            recommendations += """✅ LOW RISK IMAGE - APPEARS AUTHENTIC

This image shows good quality characteristics:
• High resolution and sharpness
• Low noise and minimal artifacts
• Good color distribution
• Minimal compression issues

✅ SAFE TO USE:
• Image appears authentic and high quality
• Suitable for most purposes
• Low likelihood of tampering
• Source appears reliable

🔍 TECHNICAL EXPLANATION:
High-quality images with good technical metrics are
statistically less likely to be tampered with because:
- Tampering often introduces quality degradation
- Legitimate sources typically provide better quality
- Professional content maintains higher standards
"""
        
        recommendations += f"""

📈 QUALITY IMPROVEMENT TIPS:
• For best results, use images with resolution > 1280×720
• Avoid heavily compressed JPEG images
• Look for images with good sharpness and minimal blur
• Prefer images from reputable sources
• Be cautious with images showing unusual quality patterns

🛠️ FOR DEVELOPERS:
This system analyzes {len(result['quality_metrics'])} different quality metrics
to provide a comprehensive assessment. The algorithm weights
blur and sharpness most heavily (25% each) as these are
strong indicators of image authenticity and quality.
"""
        
        self.recommendations_text.insert(tk.END, recommendations)
    
    def highlight_verdict(self):
        """Add color coding to the verdict in summary"""
        content = self.summary_text.get(1.0, tk.END)
        if "Not Tampered" in content:
            # Green for authentic
            self.summary_text.tag_add("authentic", "4.0", "4.end")
            self.summary_text.tag_config("authentic", foreground="#27ae60", font=("Arial", 12, "bold"))
        elif "Likely Tampered" in content or "Possibly Tampered" in content:
            # Red for tampered
            self.summary_text.tag_add("tampered", "4.0", "4.end")
            self.summary_text.tag_config("tampered", foreground="#e74c3c", font=("Arial", 12, "bold"))
        else:
            # Orange for uncertain
            self.summary_text.tag_add("uncertain", "4.0", "4.end")
            self.summary_text.tag_config("uncertain", foreground="#f39c12", font=("Arial", 12, "bold"))
    
    def clear_results(self):
        """Clear all result text widgets"""
        self.summary_text.delete(1.0, tk.END)
        self.metrics_text.delete(1.0, tk.END)
        self.recommendations_text.delete(1.0, tk.END)
    
    def save_results(self):
        """Save analysis results to file"""
        if not self.current_result:
            messagebox.showwarning("Warning", "No results to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save analysis results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"), 
                ("Text files", "*.txt"), 
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.current_result, f, indent=2)
                else:
                    # Save as formatted text
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("IMAGE QUALITY-BASED TAMPERING ANALYSIS REPORT\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(self.summary_text.get(1.0, tk.END))
                        f.write("\n" + "=" * 60 + "\n")
                        f.write(self.metrics_text.get(1.0, tk.END))
                        f.write("\n" + "=" * 60 + "\n")
                        f.write(self.recommendations_text.get(1.0, tk.END))
                
                messagebox.showinfo("Success", f"Results saved to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not save results: {str(e)}")
    
    def update_status(self, message):
        """Update the status label"""
        self.status_label.config(text=message)
        self.root.update()

def main():
    root = tk.Tk()
    app = QualityGUIDetector(root)
    
    # Add some helpful text at startup
    startup_text = """Welcome to Quality-Based Image Tampering Detection!

This system analyzes images based on quality metrics like:
• Blur levels and sharpness
• Noise patterns  
• Compression artifacts
• Resolution quality
• Color distribution

🔍 HOW IT WORKS:
High-quality images are typically more trustworthy.
Blurry, noisy, or heavily compressed images may indicate
tampering, poor sources, or processing issues.

📁 TO GET STARTED:
1. Click 'Load Image' to select any image file
2. Click 'Analyze Quality' to run the analysis
3. Review results in the tabs on the right

✨ PERFECT FOR:
• Verifying images downloaded from Google
• Checking social media images  
• Quality assessment of any photo
• Identifying potentially manipulated content"""
    
    app.summary_text.insert(tk.END, startup_text)
    app.summary_text.tag_add("startup", "1.0", tk.END)
    app.summary_text.tag_config("startup", foreground="#7f8c8d", font=("Arial", 11))
    
    root.mainloop()

if __name__ == "__main__":
    main()
