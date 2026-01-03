import os
import random
import shutil

def organize_data(num_samples=None):
    open_eye_dir = 'mrl_dataset_raw/Open-Eyes/'
    closed_eye_dir = 'mrl_dataset_raw/Close-Eyes/'
    
    # Create target directories if they don't exist
    os.makedirs('data/train/open', exist_ok=True)
    os.makedirs('data/train/closed', exist_ok=True)
    os.makedirs('data/validation/open', exist_ok=True)
    os.makedirs('data/validation/closed', exist_ok=True)

    # Get available images
    open_eye_images = [f for f in os.listdir(open_eye_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    closed_eye_images = [f for f in os.listdir(closed_eye_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(open_eye_images)} open eye images")
    print(f"Found {len(closed_eye_images)} closed eye images")

    random.shuffle(open_eye_images)
    random.shuffle(closed_eye_images)

    if num_samples is None:
        selected_open = open_eye_images
        selected_closed = closed_eye_images
        print(f"Using ALL available images")
    else:
        selected_open = open_eye_images[:min(num_samples, len(open_eye_images))]
        selected_closed = closed_eye_images[:min(num_samples, len(closed_eye_images))]
        print(f"Using {num_samples} images per class")

    train_split = 0.8
    train_open_count = int(len(selected_open) * train_split)
    train_closed_count = int(len(selected_closed) * train_split)

    train_open = selected_open[:train_open_count]
    val_open = selected_open[train_open_count:]
    train_closed = selected_closed[:train_closed_count]
    val_closed = selected_closed[train_closed_count:]

    print(f"\nCopying {len(train_open)} open eye images to training...")
    for i, image in enumerate(train_open):
        shutil.copy(os.path.join(open_eye_dir, image), 'data/train/open')

    print(f"\nCopying {len(train_closed)} closed eye images to training...")
    for i, image in enumerate(train_closed):
        shutil.copy(os.path.join(closed_eye_dir, image), 'data/train/closed')

    print(f"\nCopying {len(val_open)} open eye images to validation...")
    for image in val_open:
        shutil.copy(os.path.join(open_eye_dir, image), 'data/validation/open')

    print(f"\nCopying {len(val_closed)} closed eye images to validation...")
    for image in val_closed:
        shutil.copy(os.path.join(closed_eye_dir, image), 'data/validation/closed')

    print("\n" + "="*50)
    print("✅ Data setup complete!")
    print(f"Training images: {len(train_open) + len(train_closed)}")
    print(f"  - Open eyes: {len(train_open)}")
    print(f"  - Closed eyes: {len(train_closed)}")
    print(f"Validation images: {len(val_open) + len(val_closed)}")
    print(f"  - Open eyes: {len(val_open)}")
    print(f"  - Closed eyes: {len(val_closed)}")
    print(f"Total: {len(selected_open) + len(selected_closed)}")
    print("="*50)

if __name__ == "__main__":
    organize_data(None)