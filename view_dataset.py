
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
    axes[0].set_title(f"Authentic\n{auth_sample['filename']}")
    axes[0].axis('off')
    
    # Show tampered
    tampered_img = Image.open(os.path.join('dataset', tampered_sample['path']))
    axes[1].imshow(tampered_img)
    axes[1].set_title(f"Tampered ({tampered_sample['tampering_type']})\n{tampered_sample['filename']}")
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
    
    print(f"\nTampering types:")
    for t_type, count in tampering_types.items():
        print(f"  {t_type}: {count}")

if __name__ == "__main__":
    view_dataset_samples()
