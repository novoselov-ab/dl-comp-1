from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import pprint
import sys
import time
import os
import shutil


IMAGE_SIZE = 224
BATCH_SIZE = 30
SOURCE_DATA_FOLDER = 'processed_train_cropped_multiple_05'
DIST_DATA_FOLDER = 'generated_train_cropped_05_224x224_multiple_10000'
RGB = True
COLOR_MODE = 'rgb' if RGB else 'grayscale'
NB_SAMPLES_PER_CLASS = 10000

def generate(source_data_dir, dist_data_dir, augment = True, count = None):
    augment_datagen = ImageDataGenerator(rescale=1./255,
                                #samplewise_center=True,
                                #samplewise_std_normalization=True,
                                #zca_whitening=False,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                #channel_shift_range=0.2,
                                rotation_range=26,
                                shear_range=0.3,
                                zoom_range=[0.7, 1.3],
                                channel_shift_range=45.0,
                                horizontal_flip=False)

    no_logo_datagen = ImageDataGenerator(rescale=1./255,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                #channel_shift_range=0.2,
                                rotation_range=5,
                                shear_range=0.2,
                                channel_shift_range=15.0,
                                horizontal_flip=True)

    dummy_datagen = ImageDataGenerator(rescale=1./255)

    if os.path.exists(dist_data_dir):
        shutil.rmtree(dist_data_dir)
    os.makedirs(dist_data_dir)

    for dirname in os.listdir(source_data_dir):
        source_dir = os.path.join(source_data_dir, dirname)

        dist_dir = os.path.join(dist_data_dir, dirname)
        os.makedirs(dist_dir)

        print('From {0} to {1}', source_dir, dist_dir)

        if not augment:
            datagen = dummy_datagen
        elif dirname == 'no-logo':
            datagen = no_logo_datagen
        else:
            datagen = augment_datagen

        train_generator = datagen.flow_from_directory(source_data_dir,
                                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),  # all images will be resized to IMAGE_SIZExIMAGE_SIZE
                                                            batch_size=1,
                                                            color_mode=COLOR_MODE,
                                                            class_mode='categorical',
                                                            classes=[dirname],
                                                            save_to_dir=dist_dir)

        total = len(train_generator.filenames) if count is None else count

        i = 0
        for _, _ in train_generator:
            i += 1
            if i >= total:
                break

generate(SOURCE_DATA_FOLDER + '/train', DIST_DATA_FOLDER + '/train', True, NB_SAMPLES_PER_CLASS)
generate(SOURCE_DATA_FOLDER + '/validation', DIST_DATA_FOLDER + '/validation', False)

