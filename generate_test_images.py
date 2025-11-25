import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import os

def create_sample_images():
    """Generate sample images for testing tampering detection"""
    
    # Create output directory if it doesn't exist
    if not os.path.exists('sample_images'):
        os.makedirs('sample_images')
    
    # 1. Create a clean, authentic image
    print("Creating authentic image...")
    width, height = 800, 600
    
    # Create a gradient background
    authentic = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            authentic[i, j] = [
                int(255 * (i / height)),  # Red gradient
                int(255 * (j / width)),   # Green gradient
                int(128 + 127 * np.sin(i * j / 10000))  # Blue pattern
            ]
    
    # Add some geometric shapes
    img_pil = Image.fromarray(authentic)
    draw = ImageDraw.Draw(img_pil)
    
    # Draw some circles and rectangles
    draw.ellipse([100, 100, 200, 200], fill=(255, 255, 0), outline=(0, 0, 0))
    draw.rectangle([300, 150, 450, 300], fill=(0, 255, 255), outline=(0, 0, 0))
    draw.polygon([(500, 100), (600, 200), (550, 250), (450, 200)], fill=(255, 0, 255))
    
    # Add some text
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        draw.text((50, 400), "AUTHENTIC IMAGE - NO TAMPERING", fill=(255, 255, 255), font=font)
    except:
        draw.text((50, 400), "AUTHENTIC IMAGE", fill=(255, 255, 255))
    
    img_pil.save('sample_images/authentic_image.jpg', quality=95)
    
    # 2. Create a copy-move tampered image
    print("Creating copy-move tampered image...")
    copy_move_img = img_pil.copy()
    
    # Copy a region and paste it elsewhere
    region = copy_move_img.crop((100, 100, 200, 200))  # Copy the yellow circle
    copy_move_img.paste(region, (400, 400))  # Paste it elsewhere
    
    try:
        draw2 = ImageDraw.Draw(copy_move_img)
        draw2.text((50, 500), "COPY-MOVE TAMPERED", fill=(255, 0, 0))
    except:
        pass
    
    copy_move_img.save('sample_images/copy_move_tampered.jpg', quality=95)
    
    # 3. Create a spliced image (different compression artifacts)
    print("Creating spliced tampered image...")
    
    # Create two images with different compression levels
    img1 = img_pil.copy()
    img2 = Image.new('RGB', (400, 300), color=(50, 150, 50))
    draw3 = ImageDraw.Draw(img2)
    draw3.ellipse([50, 50, 150, 150], fill=(255, 100, 100))
    draw3.rectangle([200, 100, 350, 250], fill=(100, 100, 255))
    
    # Save img2 with different compression
    img2.save('temp_low_quality.jpg', quality=60)
    img2_compressed = Image.open('temp_low_quality.jpg')
    
    # Splice the compressed image into the original
    spliced_img = img1.copy()
    spliced_img.paste(img2_compressed, (400, 300))
    
    try:
        draw4 = ImageDraw.Draw(spliced_img)
        draw4.text((50, 500), "SPLICED TAMPERED", fill=(255, 0, 0))
    except:
        pass
    
    spliced_img.save('sample_images/spliced_tampered.jpg', quality=95)
    
    # Clean up temporary file
    if os.path.exists('temp_low_quality.jpg'):
        os.remove('temp_low_quality.jpg')
    
    # 4. Create an image with noise inconsistencies
    print("Creating noise tampered image...")
    noise_img_array = np.array(img_pil)
    
    # Add different noise levels to different regions
    region1 = noise_img_array[100:300, 100:300]
    noise1 = np.random.normal(0, 5, region1.shape).astype(np.int16)
    region1_noisy = np.clip(region1.astype(np.int16) + noise1, 0, 255).astype(np.uint8)
    noise_img_array[100:300, 100:300] = region1_noisy
    
    region2 = noise_img_array[350:500, 400:600]
    noise2 = np.random.normal(0, 25, region2.shape).astype(np.int16)
    region2_noisy = np.clip(region2.astype(np.int16) + noise2, 0, 255).astype(np.uint8)
    noise_img_array[350:500, 400:600] = region2_noisy
    
    noise_img = Image.fromarray(noise_img_array)
    try:
        draw5 = ImageDraw.Draw(noise_img)
        draw5.text((50, 500), "NOISE TAMPERED", fill=(255, 0, 0))
    except:
        pass
    
    noise_img.save('sample_images/noise_tampered.jpg', quality=95)
    
    # 5. Create an image with lighting inconsistencies
    print("Creating lighting tampered image...")
    lighting_img_array = np.array(img_pil)
    
    # Darken one region
    lighting_img_array[200:400, 200:500] = (lighting_img_array[200:400, 200:500] * 0.3).astype(np.uint8)
    
    # Brighten another region
    region_bright = lighting_img_array[50:200, 500:700].astype(np.float32)
    region_bright = np.clip(region_bright * 1.8, 0, 255).astype(np.uint8)
    lighting_img_array[50:200, 500:700] = region_bright
    
    lighting_img = Image.fromarray(lighting_img_array)
    try:
        draw6 = ImageDraw.Draw(lighting_img)
        draw6.text((50, 500), "LIGHTING TAMPERED", fill=(255, 0, 0))
    except:
        pass
    
    lighting_img.save('sample_images/lighting_tampered.jpg', quality=95)
    
    print("Sample images created successfully in 'sample_images' folder:")
    print("1. authentic_image.jpg - Clean, untampered image")
    print("2. copy_move_tampered.jpg - Contains copy-move forgery")
    print("3. spliced_tampered.jpg - Contains splicing with compression artifacts")
    print("4. noise_tampered.jpg - Contains noise inconsistencies")
    print("5. lighting_tampered.jpg - Contains lighting inconsistencies")

if __name__ == "__main__":
    create_sample_images()
