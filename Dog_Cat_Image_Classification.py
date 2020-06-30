import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin  = 'E:\Machine Learning', extract = True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

number_cats_tr = len(os.listdir(train_cats_dir))
print("number of training cats are ", number_cats_tr)

batch_size = 128;
epochs = 15;
IMG_HEIGHT = 150;
IMG_WIDTH = 150;

train_image_generator = ImageDataGenerator(rescale = 1./255)
validation_image_generator = ImageDataGenerator(rescale = 1./255)

train_data_gen = train_image_generator.flow_from_directory(directory = train_dir, shuffle = True, batch_size = batch_size, target_size=(IMG_HEIGHT,IMG_WIDTH), class_mode = 'binary')
val_data_gen = validation_image_generator.flow_from_directory(directory= validation_dir, batch_size = batch_size, target_size= (IMG_HEIGHT, IMG_WIDTH), class_mode= 'binary')

sample_training_images, _ = next(train_data_gen)

def plot_images(image_array):
    fig, axes = plt.subplots(1, 5, figsize = (20,20))
    axes = axes.flatten()
    for image, ax in zip(image_array, axes):
        ax.imshow(image)
    plt.tight_layout()
    plt.show()

plot_images(sample_training_images[:5])
