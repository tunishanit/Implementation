import os
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.image as tfi
import cv2

SIZE = 256

def load_image(image, SIZE):
    img = img_to_array(load_img(image))
    img = tfi.resize(img / 255., (SIZE, SIZE))
    return np.round(img, 4).astype(np.float32)

def load_images(image_paths, SIZE, mask=False, trim=None):
    if trim is not None:
        image_paths = image_paths[:trim]

    if mask:
        images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 1))
    else:
        images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 3))

    labels = []

    for i, image in enumerate(image_paths):
        img = load_image(image, SIZE)
        if mask:
            images[i] = img[:, :, :1]
        else:
            images[i] = img
            if 'benign' in image:
                labels.append(0)
            elif 'malignant' in image:
                labels.append(1)
            elif 'normal' in image:
                labels.append(2)

    labels = np.array(labels)
    return images, labels

def load_merged_masks(image_paths, SIZE):
    num_images = len(image_paths)
    merged_masks = np.zeros((num_images, SIZE, SIZE, 1), dtype=np.float32)

    for i, image in enumerate(image_paths):
        masks = glob(image.replace('.png', '*mask*.png'))
        merged_mask = np.zeros((SIZE, SIZE), dtype=np.float32)

        for mask_path in masks:
            mask = load_image(mask_path, SIZE)
            merged_mask = np.logical_or(merged_mask, mask[:, :, 0])

        merged_mask = np.expand_dims(merged_mask, axis=-1)
        merged_masks[i] = merged_mask

    return merged_masks

def load_data(root_path):
    classes = sorted(os.listdir(root_path))
    single_mask_paths = sorted([sorted(glob(root_path + name + "/*mask.png")) for name in classes])
    image_paths = []
    mask_paths = []

    for class_path in single_mask_paths:
        for path in class_path:
            img_path = path.replace('_mask', '')
            image_paths.append(img_path)
            mask_paths.append(path)

    images, labels = load_images(image_paths, SIZE)
    masks, _ = load_images(mask_paths, SIZE, mask=True)

    return images, masks, labels