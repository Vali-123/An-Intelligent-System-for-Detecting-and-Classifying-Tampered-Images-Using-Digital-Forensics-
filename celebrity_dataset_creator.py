import os
import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter
import random
import json
from urllib.parse import urlparse
import time

class CelebrityDatasetCreator:
    def __init__(self):
        self.dataset_dir = "celebrity_dataset"
        self.original_dir = os.path.join(self.dataset_dir, "original")
        self.tampered_dir = os.path.join(self.dataset_dir, "tampered")
        
        # Create directories
        os.makedirs(self.original_dir, exist_ok=True)
        os.makedirs(self.tampered_dir, exist_ok=True)
        
        # Celebrity names for searching
        self.celebrities = [
            "Leonardo DiCaprio", "Brad Pitt", "Angelina Jolie", "Jennifer Lawrence",
            "Tom Cruise", "Will Smith", "Scarlett Johansson", "Robert Downey Jr",
            "Chris Evans", "Chris Hemsworth", "Emma Stone", "Ryan Reynolds",
            "Dwayne Johnson", "Gal Gadot", "Margot Robbie", "Tom Holland",
            "Zendaya", "Ryan Gosling", "Emma Watson", "Natalie Portman",
            "Anne Hathaway", "Jennifer Aniston", "Matt Damon", "Mark Wahlberg",
            "Sandra Bullock", "Julia Roberts", "George Clooney", "Morgan Freeman",
            "Denzel Washington", "Samuel L Jackson", "Hugh Jackman", "Christian Bale",
            "Matthew McConaughey", "Benedict Cumberbatch", "Tom Hanks", "Meryl Streep",
            "Charlize Theron", "Reese Witherspoon", "Amy Adams", "Jessica Chastain",
            "Cate Blanchett", "Nicole Kidman", "Joaquin Phoenix", "Jake Gyllenhaal",
            "Michael Shannon", "Oscar Isaac", "Idris Elba", "Michael B Jordan",
            "Lupita Nyongo", "Viola Davis", "Mahershala Ali"
        ]
        
        # Sample celebrity image URLs (these are placeholder URLs - in real implementation, you'd use proper image search APIs)
        self.sample_urls = self.generate_sample_celebrity_urls()
    
    def generate_sample_celebrity_urls(self):
        """Generate sample URLs for celebrity images (placeholder implementation)"""
        # In a real implementation, you would use:
        # - Google Custom Search API
        # - Bing Image Search API
        # - Flickr API
        # - Other legitimate image sources with proper licensing
        
        # For demonstration, I'll create sample images programmatically
        return []
    
    def create_sample_celebrity_images(self):
        """Create sample celebrity-like images for demonstration"""
        print("Creating sample celebrity images...")
        
        # Create 50 original images
        for i in range(50):
            # Create a sample image (placeholder for real celebrity images)
            img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            
            # Add some facial features (simple rectangles to simulate faces)
            # Face outline
            cv2.rectangle(img, (150, 100), (250, 200), (220, 180, 160), -1)
            # Eyes
            cv2.circle(img, (170, 140), 8, (50, 50, 50), -1)
            cv2.circle(img, (230, 140), 8, (50, 50, 50), -1)
            # Nose
            cv2.line(img, (200, 150), (200, 170), (180, 150, 120), 2)
            # Mouth
            cv2.ellipse(img, (200, 185), (15, 8), 0, 0, 180, (150, 100, 100), -1)
            
            # Add some variation
            celebrity_name = self.celebrities[i % len(self.celebrities)].replace(" ", "_")
            filename = f"celebrity_{i+1:02d}_{celebrity_name}.jpg"
            filepath = os.path.join(self.original_dir, filename)
            
            cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Created original image: {filename}")
    
    def create_tampered_versions(self):
        """Create tampered versions of the original images"""
        print("Creating tampered versions...")
        
        original_files = [f for f in os.listdir(self.original_dir) if f.endswith('.jpg')]
        
        for i, filename in enumerate(original_files[:50]):  # Ensure we get exactly 50
            original_path = os.path.join(self.original_dir, filename)
            img = cv2.imread(original_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply random tampering techniques
            tampered_img = self.apply_tampering(img_rgb, i)
            
            # Save tampered version
            tampered_filename = f"tampered_{filename}"
            tampered_path = os.path.join(self.tampered_dir, tampered_filename)
            cv2.imwrite(tampered_path, cv2.cvtColor(tampered_img, cv2.COLOR_RGB2BGR))
            print(f"Created tampered image: {tampered_filename}")
    
    def apply_tampering(self, img, seed):
        """Apply various tampering techniques to an image"""
        random.seed(seed)  # For reproducible results
        tampering_type = random.choice(['copy_move', 'splicing', 'lighting_change', 'noise_addition', 'compression'])
        
        if tampering_type == 'copy_move':
            return self.apply_copy_move_forgery(img)
        elif tampering_type == 'splicing':
            return self.apply_splicing(img)
        elif tampering_type == 'lighting_change':
            return self.apply_lighting_change(img)
        elif tampering_type == 'noise_addition':
            return self.apply_noise_addition(img)
        else:  # compression
            return self.apply_compression_artifacts(img)
    
    def apply_copy_move_forgery(self, img):
        """Apply copy-move forgery"""
        h, w = img.shape[:2]
        
        # Select a random region to copy
        x1, y1 = random.randint(50, w//2), random.randint(50, h//2)
        size = random.randint(30, 80)
        x2, y2 = x1 + size, y1 + size
        
        # Ensure coordinates are within bounds
        x2 = min(x2, w - 10)
        y2 = min(y2, h - 10)
        
        # Copy the region
        region = img[y1:y2, x1:x2].copy()
        
        # Paste it in a different location
        paste_x = random.randint(10, w - size - 10)
        paste_y = random.randint(10, h - size - 10)
        
        img[paste_y:paste_y + (y2-y1), paste_x:paste_x + (x2-x1)] = region
        
        return img
    
    def apply_splicing(self, img):
        """Apply splicing by inserting foreign objects"""
        h, w = img.shape[:2]
        
        # Create a simple foreign object (rectangle)
        obj_w, obj_h = random.randint(40, 80), random.randint(40, 80)
        obj_x = random.randint(0, w - obj_w)
        obj_y = random.randint(0, h - obj_h)
        
        # Insert object with different color/texture
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (obj_x, obj_y), (obj_x + obj_w, obj_y + obj_h), color, -1)
        
        return img
    
    def apply_lighting_change(self, img):
        """Apply inconsistent lighting"""
        h, w = img.shape[:2]
        
        # Create a mask for the region to change
        mask = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
        radius = random.randint(50, 100)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Apply brightness change to the masked region
        brightness_factor = random.uniform(0.3, 2.0)
        img_float = img.astype(np.float32)
        img_float[mask > 0] *= brightness_factor
        img_float = np.clip(img_float, 0, 255)
        
        return img_float.astype(np.uint8)
    
    def apply_noise_addition(self, img):
        """Apply noise to simulate tampering"""
        # Add Gaussian noise to a random region
        h, w = img.shape[:2]
        
        # Create noise
        noise = np.random.normal(0, 25, img.shape).astype(np.int16)
        
        # Apply noise to a specific region
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(w//2, w), random.randint(h//2, h)
        
        noisy_img = img.astype(np.int16) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        # Apply only to selected region
        result = img.copy()
        result[y1:y2, x1:x2] = noisy_img[y1:y2, x1:x2]
        
        return result
    
    def apply_compression_artifacts(self, img):
        """Apply JPEG compression artifacts"""
        # Convert to PIL for JPEG compression
        pil_img = Image.fromarray(img)
        
        # Apply heavy compression
        import io
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=10)  # Very low quality
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        return np.array(compressed_img)
    
    def create_dataset_metadata(self):
        """Create metadata for the dataset"""
        metadata = {
            "dataset_info": {
                "name": "Celebrity Tampering Detection Dataset",
                "version": "1.0",
                "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_images": 100,
                "original_images": 50,
                "tampered_images": 50
            },
            "categories": {
                "original": "Unmodified celebrity images",
                "tampered": "Modified images with various tampering techniques"
            },
            "tampering_techniques": [
                "copy_move_forgery",
                "splicing",
                "lighting_change", 
                "noise_addition",
                "compression_artifacts"
            ]
        }
        
        metadata_path = os.path.join(self.dataset_dir, "dataset_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset metadata saved to: {metadata_path}")
    
    def create_complete_dataset(self):
        """Create the complete dataset"""
        print("Creating Celebrity Tampering Detection Dataset...")
        print(f"Dataset will be saved in: {self.dataset_dir}")
        
        # Step 1: Create original images
        self.create_sample_celebrity_images()
        
        # Step 2: Create tampered versions
        self.create_tampered_versions()
        
        # Step 3: Create metadata
        self.create_dataset_metadata()
        
        # Step 4: Print summary
        original_count = len([f for f in os.listdir(self.original_dir) if f.endswith('.jpg')])
        tampered_count = len([f for f in os.listdir(self.tampered_dir) if f.endswith('.jpg')])
        
        print(f"\nDataset creation completed!")
        print(f"Original images: {original_count}")
        print(f"Tampered images: {tampered_count}")
        print(f"Total images: {original_count + tampered_count}")
        
        return True

def main():
    creator = CelebrityDatasetCreator()
    creator.create_complete_dataset()

if __name__ == "__main__":
    main()
