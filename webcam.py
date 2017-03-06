from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import ReduceLROnPlateau, Callback, LambdaCallback, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, activity_l2
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

IMAGE_SIZE = 299
#BATCH_SIZE = 30
DATA_FOLDER = 'real_validation'


model = load_model('submitted_models/attempt_xception_1py20170210-203434[XCeption]Adam_lr0.001_d0.0_l20.00000_d0.0_2_38-0.95.hdf5')

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

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    X_batch = np.zeros((1,) + (IMAGE_SIZE, IMAGE_SIZE, 3))

    frame = imresize(frame, (IMAGE_SIZE, IMAGE_SIZE, 3), mode='RGB')
    frame = frame.astype(np.float32)
    frame *= 1./255
    X_batch[0] = frame
    start = time.time()
    preds = (model.predict(X_batch, batch_size=X_batch.shape[0]))
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

    N = 5
    pred = preds[0]
    pred = pred * w_array
    top_N = sorted(range(len(pred)), key=lambda i: pred[i])[-1:-(N+1):-1]
    for i, index in enumerate(top_N):
        text = classes[index] + ':' + str(round(pred[index], 3))
        color = (0, 0, 0) if invert else (255, 255, 255)
        cv2.putText(frame, text, (0, 12 + 20 * i), font, 0.4, color, 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    frame_index += 1


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()