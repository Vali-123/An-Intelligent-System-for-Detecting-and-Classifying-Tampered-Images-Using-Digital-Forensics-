import os
import cv2
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MLTamperingDetector:
    def __init__(self, dataset_path="celebrity_dataset"):
        self.dataset_path = dataset_path
        self.original_dir = os.path.join(dataset_path, "original")
        self.tampered_dir = os.path.join(dataset_path, "tampered")
        
        self.scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        
        self.feature_names = [
            'noise_variance_mean', 'noise_variance_std', 'noise_outliers_count',
            'jpeg_artifacts_mean', 'jpeg_artifacts_std', 'jpeg_outliers_count',
            'lighting_variance_mean', 'lighting_variance_std', 'lighting_outliers_count',
            'edge_density', 'edge_variance', 'edge_complexity',
            'color_histogram_chi2', 'color_variance_mean', 'color_variance_std',
            'texture_contrast', 'texture_dissimilarity', 'texture_homogeneity',
            'gradient_magnitude_mean', 'gradient_magnitude_std',
            'frequency_high_energy', 'frequency_low_energy'
        ]
    
    def extract_advanced_features(self, image_path):
        """Extract comprehensive features for tampering detection"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # 1. Noise Analysis Features
        noise_features = self._extract_noise_features(gray)
        features.extend(noise_features)
        
        # 2. JPEG Compression Features
        jpeg_features = self._extract_jpeg_features(gray)
        features.extend(jpeg_features)
        
        # 3. Lighting Consistency Features
        lighting_features = self._extract_lighting_features(image_rgb)
        features.extend(lighting_features)
        
        # 4. Edge Analysis Features
        edge_features = self._extract_edge_features(gray)
        features.extend(edge_features)
        
        # 5. Color Analysis Features
        color_features = self._extract_color_features(image_rgb)
        features.extend(color_features)
        
        # 6. Texture Features
        texture_features = self._extract_texture_features(gray)
        features.extend(texture_features)
        
        # 7. Gradient Features
        gradient_features = self._extract_gradient_features(gray)
        features.extend(gradient_features)
        
        # 8. Frequency Domain Features
        freq_features = self._extract_frequency_features(gray)
        features.extend(freq_features)
        
        return np.array(features)
    
    def _extract_noise_features(self, gray):
        """Extract noise-related features"""
        # High-pass filter for noise extraction
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise = cv2.filter2D(gray, -1, kernel)
        
        # Block-wise noise variance analysis
        block_size = 32
        h, w = gray.shape
        variances = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block_noise = noise[i:i+block_size, j:j+block_size]
                var = np.var(block_noise)
                variances.append(var)
        
        variances = np.array(variances)
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        
        # Count outliers
        outliers = np.sum(np.abs(variances - mean_var) > 2 * std_var)
        
        return [mean_var, std_var, outliers]
    
    def _extract_jpeg_features(self, gray):
        """Extract JPEG compression artifact features"""
        h, w = gray.shape
        artifacts = []
        
        # Analyze 8x8 DCT blocks
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = gray[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # High-frequency component energy
                high_freq = np.sum(np.abs(dct_block[4:, 4:]))
                artifacts.append(high_freq)
        
        artifacts = np.array(artifacts)
        mean_artifact = np.mean(artifacts)
        std_artifact = np.std(artifacts)
        
        # Count suspicious blocks
        outliers = np.sum(np.abs(artifacts - mean_artifact) > 2 * std_artifact)
        
        return [mean_artifact, std_artifact, outliers]
    
    def _extract_lighting_features(self, image_rgb):
        """Extract lighting consistency features"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Analyze brightness in regions
        region_size = 50
        h, w = l_channel.shape
        brightnesses = []
        
        for i in range(0, h - region_size, region_size):
            for j in range(0, w - region_size, region_size):
                region = l_channel[i:i+region_size, j:j+region_size]
                mean_brightness = np.mean(region)
                brightnesses.append(mean_brightness)
        
        brightnesses = np.array(brightnesses)
        overall_mean = np.mean(brightnesses)
        overall_std = np.std(brightnesses)
        
        # Count inconsistent regions
        outliers = np.sum(np.abs(brightnesses - overall_mean) > 2 * overall_std)
        
        return [overall_mean, overall_std, outliers]
    
    def _extract_edge_features(self, gray):
        """Extract edge-related features"""
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Edge pixel intensities
        edge_pixels = edges[edges > 0]
        edge_variance = np.var(edge_pixels) if len(edge_pixels) > 0 else 0
        
        # Edge complexity (number of contours)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_complexity = len(contours)
        
        return [edge_density, edge_variance, edge_complexity]
    
    def _extract_color_features(self, image_rgb):
        """Extract color distribution features"""
        # Calculate color histograms for each channel
        hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
        
        # Chi-square distance between channels (should be consistent for natural images)
        chi2_rg = cv2.compareHist(hist_r, hist_g, cv2.HISTCMP_CHISQR)
        
        # Color variance across image regions
        h, w = image_rgb.shape[:2]
        region_size = 64
        color_vars = []
        
        for i in range(0, h - region_size, region_size):
            for j in range(0, w - region_size, region_size):
                region = image_rgb[i:i+region_size, j:j+region_size]
                var_r = np.var(region[:, :, 0])
                var_g = np.var(region[:, :, 1])
                var_b = np.var(region[:, :, 2])
                color_vars.extend([var_r, var_g, var_b])
        
        color_vars = np.array(color_vars)
        
        return [chi2_rg, np.mean(color_vars), np.std(color_vars)]
    
    def _extract_texture_features(self, gray):
        """Extract texture features using simple statistical methods"""
        # Resize if too large for efficient processing
        if gray.shape[0] > 512 or gray.shape[1] > 512:
            gray = cv2.resize(gray, (512, 512))
        
        # Calculate texture measures using local binary patterns approach
        # Compute local standard deviation as texture measure
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Local mean
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Local variance (texture measure)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        # Texture contrast (difference between max and min in local regions)
        local_max = cv2.dilate(gray, np.ones((kernel_size, kernel_size), np.uint8))
        local_min = cv2.erode(gray, np.ones((kernel_size, kernel_size), np.uint8))
        contrast = local_max.astype(np.float32) - local_min.astype(np.float32)
        
        # Calculate statistics
        texture_contrast = np.mean(contrast)
        texture_variance = np.mean(local_var)
        texture_homogeneity = 1.0 / (1.0 + np.mean(local_var))  # Inverse of variance for homogeneity
        
        return [texture_contrast, texture_variance, texture_homogeneity]
    
    def _extract_gradient_features(self, gray):
        """Extract gradient-based features"""
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return [np.mean(magnitude), np.std(magnitude)]
    
    def _extract_frequency_features(self, gray):
        """Extract frequency domain features"""
        # FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # High frequency energy (corners)
        high_freq_mask = np.ones((h, w))
        high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 0
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
        
        # Low frequency energy (center)
        low_freq_mask = 1 - high_freq_mask
        low_freq_energy = np.sum(magnitude_spectrum * low_freq_mask)
        
        return [high_freq_energy, low_freq_energy]
    
    def load_dataset(self):
        """Load and extract features from the dataset"""
        print("Loading dataset and extracting features...")
        
        features = []
        labels = []
        filenames = []
        
        # Load original images (label = 0)
        original_files = [f for f in os.listdir(self.original_dir) if f.endswith('.jpg')]
        for filename in original_files:
            filepath = os.path.join(self.original_dir, filename)
            feature_vector = self.extract_advanced_features(filepath)
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(0)  # 0 = original
                filenames.append(filename)
                print(f"Processed original: {filename}")
        
        # Load tampered images (label = 1)
        tampered_files = [f for f in os.listdir(self.tampered_dir) if f.endswith('.jpg')]
        for filename in tampered_files:
            filepath = os.path.join(self.tampered_dir, filename)
            feature_vector = self.extract_advanced_features(filepath)
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(1)  # 1 = tampered
                filenames.append(filename)
                print(f"Processed tampered: {filename}")
        
        return np.array(features), np.array(labels), filenames
    
    def train_models(self):
        """Train machine learning models"""
        print("Training machine learning models...")
        
        # Load dataset
        X, y, filenames = self.load_dataset()
        
        print(f"Dataset loaded: {len(X)} samples, {len(self.feature_names)} features")
        print(f"Original images: {np.sum(y == 0)}")
        print(f"Tampered images: {np.sum(y == 1)}")
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("Training Random Forest...")
        self.rf_model.fit(X_train_scaled, y_train)
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        # Train SVM
        print("Training SVM...")
        self.svm_model.fit(X_train_scaled, y_train)
        svm_pred = self.svm_model.predict(X_test_scaled)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        
        # Print results
        print(f"\nModel Performance:")
        print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
        print(f"SVM Accuracy: {svm_accuracy:.3f}")
        
        # Detailed classification report
        print(f"\nRandom Forest Classification Report:")
        print(classification_report(y_test, rf_pred, target_names=['Original', 'Tampered']))
        
        print(f"\nSVM Classification Report:")
        print(classification_report(y_test, svm_pred, target_names=['Original', 'Tampered']))
        
        # Feature importance (Random Forest)
        feature_importance = self.rf_model.feature_importances_
        importance_df = list(zip(self.feature_names, feature_importance))
        importance_df.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(importance_df[:10]):
            print(f"{i+1}. {feature}: {importance:.4f}")
        
        # Save models
        self.save_models()
        
        # Create visualizations
        self.create_evaluation_plots(y_test, rf_pred, svm_pred, feature_importance)
        
        return {
            'rf_accuracy': rf_accuracy,
            'svm_accuracy': svm_accuracy,
            'feature_importance': importance_df
        }
    
    def save_models(self):
        """Save trained models and scaler"""
        models_dir = "trained_models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save models
        with open(os.path.join(models_dir, "rf_model.pkl"), 'wb') as f:
            pickle.dump(self.rf_model, f)
        
        with open(os.path.join(models_dir, "svm_model.pkl"), 'wb') as f:
            pickle.dump(self.svm_model, f)
        
        with open(os.path.join(models_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Models saved to {models_dir}/")
    
    def load_models(self):
        """Load pre-trained models"""
        models_dir = "trained_models"
        
        try:
            with open(os.path.join(models_dir, "rf_model.pkl"), 'rb') as f:
                self.rf_model = pickle.load(f)
            
            with open(os.path.join(models_dir, "svm_model.pkl"), 'rb') as f:
                self.svm_model = pickle.load(f)
            
            with open(os.path.join(models_dir, "scaler.pkl"), 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("Models loaded successfully!")
            return True
        except FileNotFoundError:
            print("No pre-trained models found. Please train the models first.")
            return False
    
    def predict_image(self, image_path):
        """Predict if an image is tampered using trained models"""
        # Extract features
        features = self.extract_advanced_features(image_path)
        if features is None:
            return {"error": "Could not process image"}
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make predictions
        rf_pred = self.rf_model.predict(features_scaled)[0]
        rf_prob = self.rf_model.predict_proba(features_scaled)[0]
        
        svm_pred = self.svm_model.predict(features_scaled)[0] 
        svm_prob = self.svm_model.predict_proba(features_scaled)[0]
        
        # Ensemble prediction (average probabilities)
        ensemble_prob = (rf_prob + svm_prob) / 2
        ensemble_pred = 1 if ensemble_prob[1] > 0.5 else 0
        
        result = {
            "image_path": image_path,
            "predictions": {
                "random_forest": {
                    "prediction": "Tampered" if rf_pred == 1 else "Original",
                    "confidence": float(max(rf_prob)),
                    "tampered_probability": float(rf_prob[1])
                },
                "svm": {
                    "prediction": "Tampered" if svm_pred == 1 else "Original", 
                    "confidence": float(max(svm_prob)),
                    "tampered_probability": float(svm_prob[1])
                },
                "ensemble": {
                    "prediction": "Tampered" if ensemble_pred == 1 else "Original",
                    "confidence": float(max(ensemble_prob)),
                    "tampered_probability": float(ensemble_prob[1])
                }
            },
            "recommendation": "Tampered" if ensemble_pred == 1 else "Original"
        }
        
        return result
    
    def create_evaluation_plots(self, y_test, rf_pred, svm_pred, feature_importance):
        """Create evaluation visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ML Tampering Detection Model Evaluation', fontsize=16)
        
        # Confusion Matrix - Random Forest
        cm_rf = confusion_matrix(y_test, rf_pred)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Original', 'Tampered'], 
                   yticklabels=['Original', 'Tampered'], ax=axes[0,0])
        axes[0,0].set_title('Random Forest Confusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # Confusion Matrix - SVM
        cm_svm = confusion_matrix(y_test, svm_pred)
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['Original', 'Tampered'], 
                   yticklabels=['Original', 'Tampered'], ax=axes[0,1])
        axes[0,1].set_title('SVM Confusion Matrix')
        axes[0,1].set_ylabel('True Label')
        axes[0,1].set_xlabel('Predicted Label')
        
        # Feature Importance
        top_features = self.feature_names[:10]  # Top 10 features
        top_importance = feature_importance[:10]
        
        axes[1,0].barh(range(len(top_features)), top_importance)
        axes[1,0].set_yticks(range(len(top_features)))
        axes[1,0].set_yticklabels([name.replace('_', ' ').title() for name in top_features])
        axes[1,0].set_xlabel('Feature Importance')
        axes[1,0].set_title('Top 10 Feature Importance (Random Forest)')
        
        # Model Comparison
        models = ['Random Forest', 'SVM']
        accuracies = [accuracy_score(y_test, rf_pred), accuracy_score(y_test, svm_pred)]
        
        axes[1,1].bar(models, accuracies, color=['skyblue', 'lightgreen'])
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_title('Model Performance Comparison')
        axes[1,1].set_ylim(0, 1)
        
        # Add accuracy values on bars
        for i, acc in enumerate(accuracies):
            axes[1,1].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('ml_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Evaluation plots saved as 'ml_model_evaluation.png'")

def main():
    detector = MLTamperingDetector()
    
    print("=== ML-Based Celebrity Image Tampering Detection ===")
    print("1. Training models on celebrity dataset...")
    
    # Train models
    results = detector.train_models()
    
    print("\n=== Testing with sample images ===")
    
    # Test with a few sample images
    test_original = os.path.join(detector.original_dir, "celebrity_01_Leonardo_DiCaprio.jpg")
    test_tampered = os.path.join(detector.tampered_dir, "tampered_celebrity_01_Leonardo_DiCaprio.jpg")
    
    if os.path.exists(test_original):
        print(f"\nTesting Original Image: {test_original}")
        result_orig = detector.predict_image(test_original)
        print(f"Prediction: {result_orig['recommendation']}")
        print(f"Ensemble Confidence: {result_orig['predictions']['ensemble']['confidence']:.3f}")
    
    if os.path.exists(test_tampered):
        print(f"\nTesting Tampered Image: {test_tampered}")  
        result_tamp = detector.predict_image(test_tampered)
        print(f"Prediction: {result_tamp['recommendation']}")
        print(f"Ensemble Confidence: {result_tamp['predictions']['ensemble']['confidence']:.3f}")
    
    print("\n=== Model Training Complete ===")
    print("You can now use the trained models to detect tampering in new images!")

if __name__ == "__main__":
    main()
