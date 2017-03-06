from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ReduceLROnPlateau, Callback, LambdaCallback, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, activity_l2
import numpy as np
import matplotlib.pyplot as plt
import pprint
import sys
import os
import time
import pickle
import random

#NB_CLASSES = 33
IMAGE_SIZE = 224
BATCH_SIZE = 16
DATA_FOLDER = 'processed_train_cropped_bbox' #generated_train_cropped_224x224_multiple'#'real_validation'
RGB = True
COLOR_MODE = 'rgb' if RGB else 'grayscale'
PLOT_BARS = False

MODEL_NAME = 'attempt_Epy20170118-002137[RMSprop_lr0005_d0_l20002_d5]69-0.73.hdf5'

model = load_model('saved_models/' + MODEL_NAME)

def learn():
    train_dir = DATA_FOLDER + '/train'

    bbox_dict = pickle.load(open(os.path.join(DATA_FOLDER, 'bbox_dict.pickle'), 'rb'))
    #pprint.pprint(bbox_dict)

    files = []
    dirs = os.listdir(train_dir)
    for dirname in dirs:
        if dirname != 'no-logo':
            files += [os.path.join(dirname, x) for x in os.listdir(os.path.join(train_dir, dirname))]

    for l in model.layers:
        l.trainable = False

    regularizer = None#l2(0.0002)

    x = Convolution2D(64, 5, 5, W_regularizer=regularizer, border_mode='same', name='conv100')(model.get_layer('conv10').output)
    x = BatchNormalization(name='batch_norm100')(x)
    x = Activation('relu', name='activation104')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool100')(x)

    x = Convolution2D(64, 5, 5, W_regularizer=regularizer, border_mode='same', name='conv101')(x)
    x = BatchNormalization(name='batch_norm101')(x)
    x = Activation('relu', name='activation103')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool101')(x)

    x = Convolution2D(64, 5, 5, W_regularizer=regularizer, border_mode='same', name='conv105')(x)
    x = BatchNormalization(name='batch_norm105')(x)
    x = Activation('relu', name='activation105')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool105')(x)
    
    x = Flatten()(x)
    x = Dense(128, W_regularizer=regularizer)(x)
    x = Activation('relu', name='activation101')(x)
    x = Dense(128, W_regularizer=regularizer)(x)
    x = Activation('relu', name='activation102')(x)
    x = Dense(4, W_regularizer=regularizer)(x)
    new_model = Model(input=model.input, output=x)

    new_model.compile(loss='mean_squared_error',
            optimizer=RMSprop(lr=0.001),
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    new_model.summary()

    X_batch = np.zeros((BATCH_SIZE,) + (IMAGE_SIZE, IMAGE_SIZE, 3))
    Y_batch = np.zeros((BATCH_SIZE,) + (4,))
    batches = 0
    while True:
        for i in range(BATCH_SIZE):
            file = random.choice(files)
            img = load_img(os.path.join(train_dir, file), grayscale=False, target_size=None)
            aspect = img.size[0] / img.size[1]
            x_factor = 1. / aspect if aspect > 1. else 1.
            y_factor = aspect if aspect < 1. else 1.
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = img_to_array(img)
            img *= 1./255
            X_batch[i] = img

            fname = os.path.split(file)[1]
            #print(fname)
            if fname in bbox_dict:
                bbox = list(bbox_dict[fname])
                bbox[0] *= x_factor
                bbox[1] *= y_factor
                bbox[2] *= x_factor
                bbox[3] *= y_factor
                Y_batch[i] = np.asarray(bbox)
            else:
                print('error')
                Y_batch[i] = np.zeros(4)

        #pprint.pprint(Y_batch)
        r = new_model.train_on_batch(X_batch, Y_batch)
        pprint.pprint(r)
        batches += 1
        if batches % 100 == 0:
            pprint.pprint('saving')
            new_model.save('saved_bbox_models/' + MODEL_NAME[:-4] + '_bbox_' + str(batches) + '_' + str(r[0]) + '.hdf5')

learn()   
#generator_predictor()
        #break

#pred = model.predict_generator(
#        test_generator,
#        val_samples=1)
#
#print(pred)
#pprint.pprint(pred)

#model.save('saved_models/' + model_name + '.hdf5')

## summarize history for accuracy
#f, (ax1, ax2) = plt.subplots(2, sharex=True)
#ax1.plot(history.history['acc'])
#ax1.plot(history.history['val_acc'])
#ax1.set_title('accuracy')
#ax1.legend(['train', 'test'], loc='upper left')
### summarize history for loss
#ax2.plot(history.history['loss'])
#ax2.plot(history.history['val_loss'])
#ax2.set_title('loss')
#ax2.legend(['train', 'test'], loc='upper left')
#plt.show()


