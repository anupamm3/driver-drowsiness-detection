import os
import random
import shutil

def organize_data(num_samples=10000):
    open_eye_dir = 'mrl_dataset_raw/Open-Eyes/'
    closed_eye_dir = 'mrl_dataset_raw/Close-Eyes/'
    
    # Create target directories if they don't exist
    os.makedirs('data/train/open', exist_ok=True)
    os.makedirs('data/train/closed', exist_ok=True)

    # Get available images
    open_eye_images = [f for f in os.listdir(open_eye_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    closed_eye_images = [f for f in os.listdir(closed_eye_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(open_eye_images)} open eye images")
    print(f"Found {len(closed_eye_images)} closed eye images")

    random.shuffle(open_eye_images)
    random.shuffle(closed_eye_images)
    selected_open = open_eye_images[:min(num_samples, len(open_eye_images))]
    selected_closed = closed_eye_images[:min(num_samples, len(closed_eye_images))]

    print(f"\nCopying {len(selected_open)} open eye images...")
    for i, image in enumerate(selected_open):
        shutil.copy(os.path.join(open_eye_dir, image), 'data/train/open')

    print(f"\nCopying {len(selected_closed)} closed eye images...")
    for i, image in enumerate(selected_closed):
        shutil.copy(os.path.join(closed_eye_dir, image), 'data/train/closed')

    print("\n" + "="*50)
    print("✅ Data setup complete!")
    print(f"Total images: {len(selected_open) + len(selected_closed)}")
    print(f"  - Open eyes: {len(selected_open)}")
    print(f"  - Closed eyes: {len(selected_closed)}")
    print("="*50)

if __name__ == "__main__":
    organize_data(10000)