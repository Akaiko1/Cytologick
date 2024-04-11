import config
import os

images_path = os.path.join(config.DATASET_FOLDER, config.IMAGES_FOLDER)
masks_path = os.path.join(config.DATASET_FOLDER, config.MASKS_FOLDER)

images = os.listdir(images_path)
masks = os.listdir(masks_path)

print(f'Sanity check passed: {images == masks}')