from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import ReduceLROnPlateau, Callback, LambdaCallback, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt
import cv2
import sys
import logging as log
import datetime as dt
import numpy as np
from pprint import pprint
import os
import time
from time import sleep
from scipy.misc import imresize
import xception


IMAGE_SIZE = 299
#BATCH_SIZE = 30
DATA_FOLDER = 'real_validation'


model_file = 'submitted_models/attempt_xception_1py20170210-203434[XCeption]Adam_lr0.001_d0.0_l20.00000_d0.0_2_38-0.95.hdf5'

model = xception.Xception(regularizer=None, include_top=False, weights=None)
model.load_weights(model_file, by_name=True)
#model = load_model(model_file)
model.summary()
model_2 = Model(input=model.input, output=model.get_layer('block14_sepconv2_act').output)

#model = resnet.ResnetBuilder.build_resnet_50((3, IMAGE_SIZE, IMAGE_SIZE), NB_CLASSES)
#model.summary()
#model.load_weights('saved_good_models/attempt_resnet_2py20170125-193142[ResNet50]RMSprop_lr0.0002_d0.0_l20.00500_d0.5_2_14-0.89.hdf5')
#model_2 = Model(input=model.input, output=model.get_layer('convolution2d_53').output)


val_dir = DATA_FOLDER + '/validation'
dirs = os.listdir(val_dir)
classes = sorted(dirs)
pprint(classes)
nb_classes = len(classes)

log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
frame_index = 0
invert = True
delay = 0

w_array = np.ones(nb_classes)
no_logo_index = classes.index('no-logo')
w_array[no_logo_index] = 75.0
w_array = w_array / np.linalg.norm(w_array)

def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('scale','image',1,5,nothing)
cv2.createTrackbar('alpha','image',0,255,nothing)


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #frame = cv2.cvtColor(frame, cv2.CV_GRAY2BGR)

    scale = cv2.getTrackbarPos('scale','image')
    alpha = cv2.getTrackbarPos('alpha','image')
    image_size = IMAGE_SIZE * scale

    X_batch = np.zeros((1,) + (image_size, image_size, 3))

    frame = imresize(frame, (image_size, image_size, 3), mode='RGB')
    frame = frame.astype(np.float32)
    frame *= 1./255
    X_batch[0] = frame
    start = time.time()
    preds_conv = (model_2.predict(X_batch, batch_size=X_batch.shape[0]))

    average = np.sum(preds_conv[0], axis=2)
    #average = average / np.linalg.norm(average)
    average -= np.min(average)
    average /= np.max(average)
    average = average**2.0

    print(average.shape)


    if frame_index % 100 == 0:
        print('Prediction time:' + str(time.time() - start))
    #preds = np.zeros((1, nb_classes))

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('i') and delay == 0:
        invert = not invert
        delay = 10
    if delay > 0:
        delay -= 1

    font = cv2.FONT_HERSHEY_SIMPLEX

    average = cv2.resize(average,(image_size, image_size), interpolation = cv2.INTER_CUBIC)
    #average = cv2.cvtColor(average, cv2.COLOR_GRAY2BGR)
    #average = cv2.applyColorMap(frame, cv2.COLORMAP_AUTUMN)
    #frame = cv2.addWeighted(frame, 0.5, average, 0.5, 0)
    average = get_colors(average, plt.cm.jet)
    average = average[:, :, 0:3]
    average = average.astype('float32')
    a = alpha / 255.
    cv2.addWeighted(frame, a, average, 1 - a, 0, frame)
    cv2.imshow('image', frame)
    #N = 5
    #pred = preds[0]
    #pred = pred * w_array
    #top_N = sorted(range(len(pred)), key=lambda i: pred[i])[-1:-(N+1):-1]
    #for i, index in enumerate(top_N):
    #    text = classes[index] + ':' + str(round(pred[index], 3))
    #    color = (0, 0, 0) if invert else (255, 255, 255)
    #    cv2.putText(frame, text, (0, 12 + 20 * i), font, 0.4, color, 1)

    # Display the resulting frame
    #cv2.imshow('Video', frame)
    frame_index += 1


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()