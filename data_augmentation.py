import numpy as np
import cv2

def apply_clahe(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = (image * 255).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return enhanced_image.astype(np.float32) / 255.0

def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def ensure_rgb(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (256, 256))
    return np.array(image, dtype=np.float32) / 255.0

def dataAugmentation(images, masks, labels):
    images_update = []
    masks_update = []

    malignant_count = 0
    benign_count = 0
    normal_count = 0

    for image, mask, label in zip(images, masks, labels):
        image = ensure_rgb(image)
        mask = cv2.resize(mask, (256, 256))
        mask = np.array(mask, dtype=np.uint8)

        images_update.append(image)
        masks_update.append(mask)

        if label == 1:
            malignant_count += 1
            blurred_image = apply_gaussian_blur(image)
            images_update.append(blurred_image)
            masks_update.append(mask.copy())
            malignant_count += 1

        elif label == 0:
            benign_count += 1

        else:
            normal_count += 1
            flipped_image = cv2.flip(image, 1)
            images_update.append(flipped_image)
            masks_update.append(mask.copy())
            normal_count += 1

            contrast_image = apply_clahe(image)
            images_update.append(contrast_image)
            masks_update.append(mask.copy())
            normal_count += 1

    print(f"Malignant count after augmentation: {malignant_count}")
    print(f"Benign count after augmentation: {benign_count}")
    print(f"Normal count after augmentation: {normal_count}")
    print(f"Total Images After Augmentation: {len(images_update)}")

    return np.array(images_update, dtype=np.float32), np.array(masks_update, dtype=np.uint8)
