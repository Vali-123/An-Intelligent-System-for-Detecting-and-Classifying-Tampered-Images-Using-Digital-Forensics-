import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
from ml_tampering_detector import MLTamperingDetector
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RealTimeTamperingDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Celebrity Image Tampering Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize the ML detector
        self.detector = MLTamperingDetector()
        self.models_loaded = False
        
        # Variables
        self.current_image_path = None
        self.current_result = None
        
        self.setup_ui()
        self.load_models()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main title
        title_label = tk.Label(
            self.root, 
            text="Real-Time Celebrity Image Tampering Detection System", 
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=10)
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel for controls
        control_frame = tk.Frame(main_frame, bg='#ecf0f1', relief=tk.RAISED, bd=2)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Control buttons
        tk.Label(control_frame, text="Controls", font=("Arial", 14, "bold"), 
                bg='#ecf0f1', fg='#2c3e50').pack(pady=10)
        
        self.load_btn = tk.Button(
            control_frame, 
            text="Load Image", 
            command=self.load_image,
            font=("Arial", 12),
            bg='#3498db',
            fg='white',
            width=15,
            height=2
        )
        self.load_btn.pack(pady=5)
        
        self.analyze_btn = tk.Button(
            control_frame, 
            text="Analyze Image", 
            command=self.analyze_image,
            font=("Arial", 12),
            bg='#e74c3c',
            fg='white',
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(pady=5)
        
        self.train_btn = tk.Button(
            control_frame, 
            text="Train Models", 
            command=self.train_models,
            font=("Arial", 12),
            bg='#f39c12',
            fg='white',
            width=15,
            height=2
        )
        self.train_btn.pack(pady=5)
        
        self.save_result_btn = tk.Button(
            control_frame, 
            text="Save Results", 
            command=self.save_results,
            font=("Arial", 12),
            bg='#27ae60',
            fg='white',
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.save_result_btn.pack(pady=5)
        
        # Status label
        tk.Label(control_frame, text="Status", font=("Arial", 12, "bold"), 
                bg='#ecf0f1', fg='#2c3e50').pack(pady=(20, 5))
        
        self.status_label = tk.Label(
            control_frame, 
            text="Ready to load image", 
            font=("Arial", 10),
            bg='#ecf0f1',
            fg='#7f8c8d',
            wraplength=150
        )
        self.status_label.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            control_frame, 
            mode='indeterminate',
            length=150
        )
        self.progress.pack(pady=5)
        
        # Right panel for image and results
        display_frame = tk.Frame(main_frame, bg='#f0f0f0')
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display frame
        image_frame = tk.Frame(display_frame, bg='#ffffff', relief=tk.RAISED, bd=2)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        tk.Label(image_frame, text="Image Preview", font=("Arial", 12, "bold"), 
                bg='#ffffff', fg='#2c3e50').pack(pady=5)
        
        self.image_label = tk.Label(
            image_frame, 
            text="No image loaded", 
            bg='#ffffff',
            fg='#95a5a6',
            font=("Arial", 12)
        )
        self.image_label.pack(expand=True)
        
        # Results frame
        results_frame = tk.Frame(display_frame, bg='#ffffff', relief=tk.RAISED, bd=2)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(results_frame, text="Analysis Results", font=("Arial", 12, "bold"), 
                bg='#ffffff', fg='#2c3e50').pack(pady=5)
        
        # Results text area
        self.results_text = tk.Text(
            results_frame, 
            height=15, 
            width=50, 
            font=("Courier", 10),
            bg='#f8f9fa',
            fg='#2c3e50'
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar for results
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
    
    def load_models(self):
        """Load pre-trained models"""
        self.update_status("Loading models...")
        try:
            if self.detector.load_models():
                self.models_loaded = True
                self.update_status("Models loaded successfully!")
            else:
                self.models_loaded = False
                self.update_status("No trained models found. Please train first.")
        except Exception as e:
            self.models_loaded = False
            self.update_status(f"Error loading models: {str(e)}")
    
    def load_image(self):
        """Load an image file"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=file_types
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.config(state=tk.NORMAL)
            self.update_status(f"Image loaded: {os.path.basename(file_path)}")
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.save_result_btn.config(state=tk.DISABLED)
    
    def display_image(self, image_path):
        """Display the loaded image"""
        try:
            # Load and resize image for display
            pil_image = Image.open(image_path)
            
            # Resize to fit in the display area
            display_size = (400, 300)
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update the label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {str(e)}")
    
    def analyze_image(self):
        """Analyze the loaded image for tampering"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Models not loaded! Please train the models first.")
            return
        
        self.update_status("Analyzing image...")
        self.progress.start()
        
        # Run analysis in a separate thread to avoid freezing UI
        self.root.after(100, self._perform_analysis)
    
    def _perform_analysis(self):
        """Perform the actual analysis"""
        try:
            # Analyze the image
            result = self.detector.predict_image(self.current_image_path)
            self.current_result = result
            
            # Display results
            self.display_results(result)
            
            self.progress.stop()
            self.update_status("Analysis complete!")
            self.save_result_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.progress.stop()
            self.update_status(f"Analysis failed: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def display_results(self, result):
        """Display analysis results"""
        self.results_text.delete(1.0, tk.END)
        
        if "error" in result:
            self.results_text.insert(tk.END, f"Error: {result['error']}\n")
            return
        
        # Format and display results
        output = f"TAMPERING DETECTION RESULTS\n"
        output += f"=" * 50 + "\n\n"
        output += f"Image: {os.path.basename(result['image_path'])}\n\n"
        
        # Overall recommendation
        recommendation = result['recommendation']
        confidence = result['predictions']['ensemble']['confidence']
        tampered_prob = result['predictions']['ensemble']['tampered_probability']
        
        output += f"FINAL VERDICT: {recommendation}\n"
        output += f"Confidence: {confidence:.3f}\n"
        output += f"Tampering Probability: {tampered_prob:.3f}\n\n"
        
        # Individual model predictions
        output += f"DETAILED PREDICTIONS:\n"
        output += f"-" * 30 + "\n"
        
        for model_name, pred_data in result['predictions'].items():
            model_display = model_name.replace('_', ' ').title()
            output += f"\n{model_display}:\n"
            output += f"  Prediction: {pred_data['prediction']}\n"
            output += f"  Confidence: {pred_data['confidence']:.3f}\n"
            output += f"  Tampered Prob: {pred_data['tampered_probability']:.3f}\n"
        
        # Risk assessment
        output += f"\nRISK ASSESSMENT:\n"
        output += f"-" * 20 + "\n"
        if tampered_prob > 0.8:
            risk_level = "VERY HIGH"
            risk_color = "red"
        elif tampered_prob > 0.6:
            risk_level = "HIGH"
            risk_color = "orange"
        elif tampered_prob > 0.4:
            risk_level = "MEDIUM"
            risk_color = "yellow"
        else:
            risk_level = "LOW"
            risk_color = "green"
        
        output += f"Risk Level: {risk_level}\n"
        
        if recommendation == "Tampered":
            output += f"\n⚠️  WARNING: This image shows signs of tampering!\n"
            output += f"Recommended action: Verify authenticity before use.\n"
        else:
            output += f"\n✅ This image appears to be authentic.\n"
        
        self.results_text.insert(tk.END, output)
    
    def train_models(self):
        """Train the machine learning models"""
        self.update_status("Training models...")
        self.progress.start()
        
        # Run training in a separate thread
        self.root.after(100, self._perform_training)
    
    def _perform_training(self):
        """Perform the actual model training"""
        try:
            # Train models
            results = self.detector.train_models()
            
            self.progress.stop()
            self.models_loaded = True
            self.update_status("Models trained successfully!")
            
            # Show training results
            messagebox.showinfo(
                "Training Complete", 
                f"Models trained successfully!\n\n"
                f"Random Forest Accuracy: {results['rf_accuracy']:.3f}\n"
                f"SVM Accuracy: {results['svm_accuracy']:.3f}"
            )
            
        except Exception as e:
            self.progress.stop()
            self.update_status(f"Training failed: {str(e)}")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def save_results(self):
        """Save analysis results to file"""
        if not self.current_result:
            messagebox.showwarning("Warning", "No results to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.current_result, f, indent=2)
                else:
                    # Save as text
                    with open(file_path, 'w') as f:
                        f.write(self.results_text.get(1.0, tk.END))
                
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not save results: {str(e)}")
    
    def update_status(self, message):
        """Update the status label"""
        self.status_label.config(text=message)
        self.root.update()

def main():
    root = tk.Tk()
    app = RealTimeTamperingDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
