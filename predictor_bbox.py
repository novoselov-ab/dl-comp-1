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

#NB_CLASSES = 33
IMAGE_SIZE = 224
#BATCH_SIZE = 30
DATA_FOLDER = 'generated_train_cropped_multiple_2_224x224_bbox_2000'
RGB = True
COLOR_MODE = 'rgb' if RGB else 'grayscale'
PLOT = True

model = load_model('saved_bbox_models/attempt_bbox_1py20170120-191911[no_tag]RMSprop_lr0.001_d0.0_l20.00500_d0.507-0.17.hdf5')
model.summary()

bbox_dict = pickle.load(open(os.path.join(DATA_FOLDER, 'bbox_dict.pickle'), 'rb'))

def bbox_to_patch(bbox, color='r'):
    (by0, by1, bx0, bx1) = bbox
    bw = bx1 - bx0
    bh = by1 - by0
    return patches.Rectangle((bx0 * IMAGE_SIZE, by0 * IMAGE_SIZE), bw * IMAGE_SIZE, bh * IMAGE_SIZE, linewidth=1, edgecolor=color, facecolor='none')
    

def evaluate_on_file(filepath, class_index, classes):
    nb_classes = len(classes)

    img = load_img(filepath,
                grayscale=False,
                target_size=None)

    fig = plt.figure(figsize=(10, 10)) if PLOT else None
    plts = 1

    X_batch = np.zeros((1,) + (IMAGE_SIZE, IMAGE_SIZE, 3))

    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    x = img_to_array(img)
    x *= 1./255
    X_batch[0] = x

    pred_label, pred_bbox = (model.predict(X_batch, batch_size=X_batch.shape[0]))
    #pprint.pprint(pred_bbox)

    if fig != None:
        ax = fig.add_subplot(2, 1, plts)
        plts += 1
        ax.imshow(X_batch[0])

        bbox_count = int(len(pred_bbox[0]) / 4)
        print(bbox_count)
        for c in range(bbox_count):
            (by0, by1, bx0, bx1) = (pred_bbox[0][c * 4], pred_bbox[0][c * 4 + 1], pred_bbox[0][c * 4 + 2], pred_bbox[0][c * 4 + 3])
            if by1 > by0 and bx1 > bx0:
                print ('Got something!')
                ax.add_patch(bbox_to_patch((by0, by1, bx0, bx1)))

        f = os.path.split(filepath)[1]
        if f in bbox_dict:
            ax.add_patch(bbox_to_patch(bbox_dict[f], 'g'))

        ax = fig.add_subplot(2, 1, plts)
        plts += 1
        width = 5
        ax.bar(list(range(nb_classes)), pred_label[0], 1.0)
        ax.set_xticks([x + 0.5 for x in range(nb_classes)])
        ax.set_xticklabels(classes, size=7, rotation=80)

        plt.show()

    if fig != None:
        plt.close(fig)
 

def manual_predictor():
    val_dir = DATA_FOLDER + '/train'
    true_index = 0
    dirs = os.listdir(val_dir)
    correct = 0
    correct_per_class = np.zeros(len(dirs))
    total = 0
    random.shuffle(dirs)
    for dirname in dirs:
        from_dir = os.path.join(val_dir, dirname)

        files = os.listdir(from_dir)
        random.shuffle(files)
        for file in files:
            evaluate_on_file(os.path.join(from_dir, file), true_index, dirs)
            break #hehe

manual_predictor()   
