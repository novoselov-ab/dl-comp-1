from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ReduceLROnPlateau, Callback, LambdaCallback, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, activity_l2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pprint
import sys
import random
import pickle
import os
import time

IMAGE_SIZE = 224
DATA_FOLDER = 'generated_train_cropped_multiple_2_224x224_bbox_1'
RGB = True
PLOT = True

bbox_dict = pickle.load(open(os.path.join(DATA_FOLDER, 'bbox_dict.pickle'), 'rb'))

def evaluate_on_file(filepath, classes):
    img = load_img(filepath,
                grayscale=False,
                target_size=None)

    fig = plt.figure(figsize=(20, 20)) if PLOT else None
    plts = 1

    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    x = img_to_array(img)
    x *= 1./255

    print(filepath)

    if fig != None:
        ax = fig.add_subplot(1, 1, plts)
        plts += 1
        ax.imshow(x)

        f = os.path.split(filepath)[1]
        if f in bbox_dict:
            print(bbox_dict[f])
            (by0, by1, bx0, bx1) = bbox_dict[f]
            bw = bx1 - bx0
            bh = by1 - by0
            rect = patches.Rectangle((bx0 * IMAGE_SIZE, by0 * IMAGE_SIZE), bw * IMAGE_SIZE, bh * IMAGE_SIZE, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        plt.show()

    if fig != None:
        plt.close(fig)
 

def do_viz():
    val_dir = DATA_FOLDER + '/train'
    dirs = os.listdir(val_dir)
    random.shuffle(dirs)
    for dirname in dirs:
        from_dir = os.path.join(val_dir, dirname)

        files = os.listdir(from_dir)
        random.shuffle(files)
        for file in files:
            evaluate_on_file(os.path.join(from_dir, file), dirs)
            break #hehe

do_viz()   
