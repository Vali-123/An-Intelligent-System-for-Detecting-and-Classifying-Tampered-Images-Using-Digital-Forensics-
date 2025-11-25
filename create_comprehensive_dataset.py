import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import cv2
import os
import random
import json
from datetime import datetime
import requests
from io import BytesIO

def create_dataset_structure():
    """Create directory structure for the dataset"""
    base_dirs = [
        'dataset/authentic',
        'dataset/tampered/copy_move',
        'dataset/tampered/splicing',
        'dataset/tampered/noise_injection',
        'dataset/tampered/lighting_modification',
        'dataset/tampered/compression_artifacts',
        'dataset/metadata'
    ]
    
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def generate_synthetic_face(width=400, height=400):
    """Generate a synthetic face-like image"""
    # Create base skin tone
    skin_colors = [
        (255, 220, 177),  # Light skin
        (241, 194, 125),  # Medium skin
        (224, 172, 105),  # Olive skin
        (198, 134, 66),   # Brown skin
        (161, 102, 94),   # Dark skin
    ]
    
    skin_color = random.choice(skin_colors)
    img = Image.new('RGB', (width, height), skin_color)
    draw = ImageDraw.Draw(img)
    
    # Face outline (oval)
    face_margin = 40
    draw.ellipse([face_margin, face_margin, width-face_margin, height-face_margin], 
                fill=skin_color, outline=(0,0,0), width=2)
    
    # Eyes
    eye_y = height // 3
    eye1_x = width // 4
    eye2_x = 3 * width // 4
    eye_size = 30
    
    # Eye whites
    draw.ellipse([eye1_x-eye_size//2, eye_y-15, eye1_x+eye_size//2, eye_y+15], 
                fill=(255,255,255), outline=(0,0,0))
    draw.ellipse([eye2_x-eye_size//2, eye_y-15, eye2_x+eye_size//2, eye_y+15], 
                fill=(255,255,255), outline=(0,0,0))
    
    # Pupils
    pupil_colors = [(139, 69, 19), (101, 67, 33), (34, 139, 34), (0, 0, 139)]
    pupil_color = random.choice(pupil_colors)
    draw.ellipse([eye1_x-10, eye_y-10, eye1_x+10, eye_y+10], fill=pupil_color)
    draw.ellipse([eye2_x-10, eye_y-10, eye2_x+10, eye_y+10], fill=pupil_color)
    
    # Nose
    nose_x = width // 2
    nose_y = height // 2
    draw.polygon([(nose_x-5, nose_y-10), (nose_x+5, nose_y-10), (nose_x, nose_y+10)], 
                fill=tuple(max(0, c-20) for c in skin_color))
    
    # Mouth
    mouth_y = 2 * height // 3
    draw.arc([width//2-30, mouth_y-10, width//2+30, mouth_y+10], 0, 180, fill=(139, 69, 19), width=3)
    
    # Hair
    hair_colors = [(139, 69, 19), (101, 67, 33), (0, 0, 0), (255, 255, 0), (165, 42, 42)]
    hair_color = random.choice(hair_colors)
    draw.ellipse([face_margin-10, face_margin-20, width-face_margin+10, height//2], 
                fill=hair_color)
    
    return img

def generate_synthetic_object(width=400, height=400, object_type='geometric'):
    """Generate synthetic objects for tampering"""
    img = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    if object_type == 'geometric':
        # Random geometric shapes
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for _ in range(random.randint(3, 8)):
            color = random.choice(colors)
            shape_type = random.choice(['circle', 'rectangle', 'triangle'])
            
            x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
            x2, y2 = x1 + random.randint(50, 150), y1 + random.randint(50, 150)
            
            if shape_type == 'circle':
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0,0,0))
            elif shape_type == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0,0,0))
            else:  # triangle
                draw.polygon([(x1, y2), (x2, y2), ((x1+x2)//2, y1)], fill=color, outline=(0,0,0))
    
    elif object_type == 'nature':
        # Simple landscape
        # Sky
        draw.rectangle([0, 0, width, height//2], fill=(135, 206, 235))
        # Ground
        draw.rectangle([0, height//2, width, height], fill=(34, 139, 34))
        # Sun
        draw.ellipse([width-100, 20, width-20, 100], fill=(255, 255, 0))
        # Mountains
        draw.polygon([(0, height//2), (width//3, height//4), (2*width//3, height//2)], fill=(139, 137, 137))
        # Trees
        for i in range(5):
            x = random.randint(50, width-50)
            y = height//2
            draw.rectangle([x-5, y, x+5, y+50], fill=(139, 69, 19))  # trunk
            draw.ellipse([x-20, y-20, x+20, y+20], fill=(0, 100, 0))  # leaves
    
    return img

def create_copy_move_forgery(base_image):
    """Create copy-move forgery by duplicating regions"""
    img_array = np.array(base_image)
    height, width = img_array.shape[:2]
    
    # Select random region to copy
    region_size = random.randint(50, min(width, height) // 4)
    src_x = random.randint(0, width - region_size)
    src_y = random.randint(0, height - region_size)
    
    # Select destination (ensure no overlap)
    attempts = 0
    while attempts < 10:
        dst_x = random.randint(0, width - region_size)
        dst_y = random.randint(0, height - region_size)
        
        # Check for minimal overlap
        if abs(dst_x - src_x) > region_size//2 or abs(dst_y - src_y) > region_size//2:
            break
        attempts += 1
    
    # Copy the region
    copied_region = img_array[src_y:src_y+region_size, src_x:src_x+region_size].copy()
    img_array[dst_y:dst_y+region_size, dst_x:dst_x+region_size] = copied_region
    
    return Image.fromarray(img_array)

def create_splicing_forgery(base_image, splice_image):
    """Create splicing forgery by combining two images"""
    base_array = np.array(base_image)
    splice_array = np.array(splice_image)
    
    # Resize splice image to fit
    splice_size = random.randint(100, min(base_array.shape[0], base_array.shape[1]) // 2)
    splice_resized = cv2.resize(splice_array, (splice_size, splice_size))
    
    # Random position
    max_x = base_array.shape[1] - splice_size
    max_y = base_array.shape[0] - splice_size
    
    if max_x > 0 and max_y > 0:
        pos_x = random.randint(0, max_x)
        pos_y = random.randint(0, max_y)
        
        # Create a soft mask for blending
        mask = np.ones((splice_size, splice_size, 3), dtype=np.float32)
        
        # Soften edges
        border_size = 10
        for i in range(border_size):
            alpha = i / border_size
            mask[i, :] *= alpha
            mask[-i-1, :] *= alpha
            mask[:, i] *= alpha
            mask[:, -i-1] *= alpha
        
        # Blend images
        result = base_array.copy().astype(np.float32)
        splice_region = splice_resized.astype(np.float32)
        
        for c in range(3):
            result[pos_y:pos_y+splice_size, pos_x:pos_x+splice_size, c] = (
                result[pos_y:pos_y+splice_size, pos_x:pos_x+splice_size, c] * (1 - mask[:, :, c]) +
                splice_region[:, :, c] * mask[:, :, c]
            )
        
        return Image.fromarray(result.astype(np.uint8))
    
    return base_image

def add_noise_tampering(base_image):
    """Add inconsistent noise to simulate tampering"""
    img_array = np.array(base_image).astype(np.float32)
    height, width = img_array.shape[:2]
    
    # Add different noise levels to different regions
    num_regions = random.randint(2, 5)
    
    for _ in range(num_regions):
        # Random region
        region_size = random.randint(80, 200)
        x = random.randint(0, max(1, width - region_size))
        y = random.randint(0, max(1, height - region_size))
        
        # Random noise intensity
        noise_intensity = random.uniform(5, 30)
        noise = np.random.normal(0, noise_intensity, 
                                (min(region_size, height-y), min(region_size, width-x), 3))
        
        img_array[y:y+noise.shape[0], x:x+noise.shape[1]] += noise
    
    img_array = np.clip(img_array, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def modify_lighting(base_image):
    """Modify lighting in certain regions"""
    img_array = np.array(base_image).astype(np.float32)
    height, width = img_array.shape[:2]
    
    # Create lighting modifications
    num_modifications = random.randint(1, 3)
    
    for _ in range(num_modifications):
        # Random region
        region_size = random.randint(100, min(width, height) // 2)
        x = random.randint(0, max(1, width - region_size))
        y = random.randint(0, max(1, height - region_size))
        
        # Random lighting change
        brightness_factor = random.uniform(0.3, 2.0)
        
        region = img_array[y:y+region_size, x:x+region_size]
        img_array[y:y+region_size, x:x+region_size] = region * brightness_factor
    
    img_array = np.clip(img_array, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def add_compression_artifacts(base_image):
    """Add compression artifacts by saving with different quality levels"""
    # Save with very low quality and reload
    temp_buffer = BytesIO()
    quality = random.randint(10, 40)
    base_image.save(temp_buffer, format='JPEG', quality=quality)
    temp_buffer.seek(0)
    
    # Reload and modify specific regions with different compression
    compressed_img = Image.open(temp_buffer)
    
    # Create a mixed compression artifact
    img_array = np.array(compressed_img)
    height, width = img_array.shape[:2]
    
    # Add blocking artifacts manually
    block_size = 8
    for y in range(0, height - block_size, block_size * 2):
        for x in range(0, width - block_size, block_size * 2):
            if random.random() < 0.3:  # 30% chance
                # Create visible block boundary
                block_region = img_array[y:y+block_size, x:x+block_size]
                mean_color = np.mean(block_region, axis=(0, 1))
                img_array[y:y+block_size, x:x+block_size] = mean_color
    
    return Image.fromarray(img_array)

def generate_comprehensive_dataset():
    """Generate a comprehensive dataset with authentic and tampered images"""
    print("🚀 Starting comprehensive dataset generation...")
    
    # Create directory structure
    create_dataset_structure()
    
    # Dataset configuration
    num_authentic = 50
    num_tampered_per_type = 10  # 5 types × 10 = 50 tampered images
    
    dataset_info = {
        "creation_date": datetime.now().isoformat(),
        "total_images": num_authentic + (num_tampered_per_type * 5),
        "authentic_count": num_authentic,
        "tampered_count": num_tampered_per_type * 5,
        "tampering_types": [
            "copy_move", "splicing", "noise_injection", 
            "lighting_modification", "compression_artifacts"
        ]
    }
    
    print(f"📊 Generating {dataset_info['total_images']} images total...")
    
    # Generate authentic images
    print("\n📷 Generating authentic images...")
    authentic_images = []
    
    for i in range(num_authentic):
        if i % 2 == 0:
            # Generate synthetic faces
            img = generate_synthetic_face(
                width=random.randint(300, 600),
                height=random.randint(300, 600)
            )
        else:
            # Generate synthetic objects/scenes
            object_type = random.choice(['geometric', 'nature'])
            img = generate_synthetic_object(
                width=random.randint(300, 600),
                height=random.randint(300, 600),
                object_type=object_type
            )
        
        # Add some natural variation
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Save authentic image
        filename = f"authentic_{i+1:03d}.jpg"
        filepath = os.path.join('dataset/authentic', filename)
        img.save(filepath, quality=random.randint(85, 95))
        authentic_images.append(img)
        
        if (i + 1) % 10 == 0:
            print(f"   ✅ Generated {i+1}/{num_authentic} authentic images")
    
    print(f"✅ Completed {num_authentic} authentic images")
    
    # Generate tampered images
    tampering_methods = [
        ("copy_move", create_copy_move_forgery),
        ("splicing", lambda img: create_splicing_forgery(img, random.choice(authentic_images))),
        ("noise_injection", add_noise_tampering),
        ("lighting_modification", modify_lighting),
        ("compression_artifacts", add_compression_artifacts)
    ]
    
    for method_name, method_func in tampering_methods:
        print(f"\n🔧 Generating {method_name} tampered images...")
        
        for i in range(num_tampered_per_type):
            # Select random authentic image as base
            base_img = random.choice(authentic_images).copy()
            
            # Apply tampering method
            if method_name == "splicing":
                # For splicing, we need a second image
                splice_source = random.choice(authentic_images)
                tampered_img = create_splicing_forgery(base_img, splice_source)
            else:
                tampered_img = method_func(base_img)
            
            # Save tampered image
            filename = f"{method_name}_{i+1:03d}.jpg"
            filepath = os.path.join(f'dataset/tampered/{method_name}', filename)
            tampered_img.save(filepath, quality=random.randint(75, 95))
            
            if (i + 1) % 5 == 0:
                print(f"   ✅ Generated {i+1}/{num_tampered_per_type} {method_name} images")
        
        print(f"✅ Completed {num_tampered_per_type} {method_name} images")
    
    # Save dataset metadata
    with open('dataset/metadata/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create labels file for the dataset
    labels = []
    
    # Add authentic image labels
    for i in range(num_authentic):
        labels.append({
            "filename": f"authentic_{i+1:03d}.jpg",
            "path": f"authentic/authentic_{i+1:03d}.jpg",
            "label": "authentic",
            "tampering_type": None
        })
    
    # Add tampered image labels
    for method_name, _ in tampering_methods:
        for i in range(num_tampered_per_type):
            labels.append({
                "filename": f"{method_name}_{i+1:03d}.jpg",
                "path": f"tampered/{method_name}/{method_name}_{i+1:03d}.jpg",
                "label": "tampered",
                "tampering_type": method_name
            })
    
    # Save labels
    with open('dataset/metadata/labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"\n🎉 Dataset generation completed!")
    print(f"📊 Total images: {len(labels)}")
    print(f"📁 Authentic: {num_authentic}")
    print(f"🔧 Tampered: {num_tampered_per_type * 5}")
    print(f"💾 Saved to: dataset/")
    print(f"📋 Labels saved to: dataset/metadata/labels.json")
    
    return dataset_info

def create_dataset_viewer():
    """Create a simple dataset viewer script"""
    viewer_code = '''
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import random

def view_dataset_samples():
    """View random samples from the dataset"""
    
    # Load labels
    with open('dataset/metadata/labels.json', 'r') as f:
        labels = json.load(f)
    
    # Get samples
    authentic_samples = [l for l in labels if l['label'] == 'authentic']
    tampered_samples = [l for l in labels if l['label'] == 'tampered']
    
    # Select random samples
    auth_sample = random.choice(authentic_samples)
    tampered_sample = random.choice(tampered_samples)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show authentic
    auth_img = Image.open(os.path.join('dataset', auth_sample['path']))
    axes[0].imshow(auth_img)
    axes[0].set_title(f"Authentic\\n{auth_sample['filename']}")
    axes[0].axis('off')
    
    # Show tampered
    tampered_img = Image.open(os.path.join('dataset', tampered_sample['path']))
    axes[1].imshow(tampered_img)
    axes[1].set_title(f"Tampered ({tampered_sample['tampering_type']})\\n{tampered_sample['filename']}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Dataset Statistics:")
    print(f"Total images: {len(labels)}")
    print(f"Authentic: {len(authentic_samples)}")
    print(f"Tampered: {len(tampered_samples)}")
    
    tampering_types = {}
    for sample in tampered_samples:
        t_type = sample['tampering_type']
        tampering_types[t_type] = tampering_types.get(t_type, 0) + 1
    
    print(f"\\nTampering types:")
    for t_type, count in tampering_types.items():
        print(f"  {t_type}: {count}")

if __name__ == "__main__":
    view_dataset_samples()
'''
    
    with open('view_dataset.py', 'w') as f:
        f.write(viewer_code)
    
    print("📊 Created dataset viewer: view_dataset.py")

if __name__ == "__main__":
    # Generate the comprehensive dataset
    dataset_info = generate_comprehensive_dataset()
    
    # Create dataset viewer
    create_dataset_viewer()
    
    print("\n🎯 Next steps:")
    print("1. Run 'python view_dataset.py' to view sample images")
    print("2. Use the dataset with your Streamlit app")
    print("3. Upload any image from dataset/ folders to test the system")
