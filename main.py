import tensorflow as tf
from sklearn.model_selection import train_test_split
from .data_loaded import load_images
from .data_augmentation import dataAugmentation
from model import build_multiresunet
from .utils import dice_coefficient, iou_metric, precision,recall, sensitivity, dice_loss,combined_loss,jaccard_coefficient
import os
import glob
from .model import build_multiresunet
import os
import keras
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
print(tf.__version__)
import tensorflow.image as tfi
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import concatenate, Conv2DTranspose, BatchNormalization, Dropout, Activation, Add
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Activation, Multiply, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape
from keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from keras.initializers import he_normal
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from .train import MetricsLogger

# Define paths
root_path = "path_of_dataset/"
save_dir = "saved_model/"
os.makedirs(save_dir, exist_ok=True)

# Load dataset
image_paths = [path.replace('_mask', '') for path in glob(root_path + "*/*mask.png")]
mask_paths = glob(root_path + "*/*mask.png")

images, labels = load_images(image_paths, 256)
masks, _ = load_images(mask_paths, 256, mask=True)

# Augment data
images_aug, masks_aug = dataAugmentation(images, masks, labels)
input_shape = (256, 256, 3)
# Split dataset
x_train, x_test, y_train, y_test = train_test_split(images_aug, masks_aug, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

model = build_multiresunet(input_shape)
model.compile(optimizer=Adam(learning_rate=1e-3), loss= combined_loss, metrics=['accuracy', dice_coefficient, precision, recall,iou_metric,sensitivity,dice_loss,binary_loss,jaccard_coefficient ])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
metrics_logger = MetricsLogger()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
filepath=os.path.join(save_dir, "best_model.h5"),
monitor='val_loss',
save_best_only=True,
mode='min',
verbose=1
)

history = model.fit(
    x_train, y_train,
    batch_size=16,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[reduce_lr, metrics_logger],
    verbose=1
)

def plot_loss(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid()

    plt.show()

# Call the function to plot loss
plot_loss(history)
