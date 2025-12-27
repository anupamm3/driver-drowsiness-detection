import os
import random
import shutil

def organize_data():
    open_eye_dir = 'mrl_dataset_raw/Open-Eyes/'
    closed_eye_dir = 'mrl_dataset_raw/Close-Eyes/'
    
    # Create target directories if they don't exist
    os.makedirs('data/train/open', exist_ok=True)
    os.makedirs('data/train/closed', exist_ok=True)

    # Get a random sample of 2000 images from Open-Eyes
    open_eye_images = os.listdir(open_eye_dir)
    random.shuffle(open_eye_images)
    selected_open_eye_images = open_eye_images[:2000]

    # Copy selected images to the training directory
    for image in selected_open_eye_images:
        shutil.copy(os.path.join(open_eye_dir, image), 'data/train/open')

    # Get a random sample of 2000 images from Close-Eyes
    closed_eye_images = os.listdir(closed_eye_dir)
    random.shuffle(closed_eye_images)
    selected_closed_eye_images = closed_eye_images[:2000]

    # Copy selected images to the training directory
    for image in selected_closed_eye_images:
        shutil.copy(os.path.join(closed_eye_dir, image), 'data/train/closed')

    print("Data setup complete")

if __name__ == "__main__":
    organize_data()