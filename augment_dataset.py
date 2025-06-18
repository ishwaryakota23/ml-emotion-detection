import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Input and output directories
input_dir = 'emotion_dataset_cleaned'
output_dir = 'emotion_dataset_augmented'

# Create augmentation generator
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# Make sure output folders exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each emotion folder
for emotion in os.listdir(input_dir):
    input_path = os.path.join(input_dir, emotion)
    output_path = os.path.join(output_dir, emotion)
    os.makedirs(output_path, exist_ok=True)

    files = os.listdir(input_path)
    print(f"🔁 Augmenting {emotion} images...")

    for idx, file in enumerate(tqdm(files)):
        try:
            img_path = os.path.join(input_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Reshape to (1, 48, 48, 1) for grayscale
            img = img.reshape((1,) + img.shape + (1,))

            # Generate 4 augmented images per original
            i = 0
            for batch in datagen.flow(img, batch_size=1):
                aug_img = batch[0].reshape((48, 48))
                save_path = os.path.join(output_path, f"{emotion}_{idx}_aug{i}.jpg")
                cv2.imwrite(save_path, aug_img)
                i += 1
                if i >= 4:
                    break

        except Exception as e:
            continue

print("🎉 Augmentation complete! New dataset in: emotion_dataset_augmented/")
