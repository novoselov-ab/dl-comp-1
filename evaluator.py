from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ReduceLROnPlateau, Callback, LambdaCallback, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, activity_l2
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pprint
import sys
import time
import tensorflow as tf
import os
from itertools import product
from functools import partial

_EPSILON = 10e-8


#NB_CLASSES = 33
IMAGE_SIZE = 299
#BATCH_SIZE = 30
#DATA_FOLDER = 'generated_train_cropped'
DATA_FOLDER = 'real_validation'
TEST_FOLDER = 'real_test'
RGB = True
COLOR_MODE = 'rgb' if RGB else 'grayscale'
PLOT_ENABLED = False
WRITE_SUBMISSION = True

#model = load_model('saved_models/attempt_Bpy20170116-002541[RMSprop_1]85-0.67.hdf5')
model = load_model('saved_models/attempt_xception_1py20170301-193816[XCeption_pretrain]Adam_lr0.0005_d0.0_l2None_d0.0_2_65-0.83.hdf5')
#model = load_model('saved_models/attempt_resnet_2py20170124-181713[ResNet50]RMSprop_lr0.001_d0.0_l20.00500_d0.5_2_27-0.88.hdf5')
model.summary()

test_datagen = ImageDataGenerator(rescale=1./255,
        #samplewise_center=True,
        #samplewise_std_normalization=True,
        zca_whitening=False)

valid_generator = test_datagen.flow_from_directory(
        DATA_FOLDER + '/validation',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=8,
        color_mode=COLOR_MODE,
        class_mode='categorical')
        #save_to_dir='debug_validation')

nb_samples = len(valid_generator.filenames)
nb_classes = len(valid_generator.class_indices)
inv_map = {v: k for k, v in valid_generator.class_indices.items()}
inv_map_list = [inv_map[x] for x in range(nb_classes)]

#def _to_tensor(x, dtype):
#    x = tf.convert_to_tensor(x)
#    if x.dtype != dtype:
#        x = tf.cast(x, dtype)
#    return x

#def w_categorical_crossentropy(output, target, weights):
#    '''Categorical crossentropy between an output tensor
#    and a target tensor, where the target is a tensor of the same
#    shape as the output.
#    '''
#    # scale preds so that the class probas of each sample sum to 1
#    output /= tf.reduce_sum(output,
#                            reduction_indices=len(output.get_shape()) - 1,
#                            keep_dims=True)
#    # manual computation of crossentropy1
#    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
#    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
#    return - tf.reduce_sum(target * tf.log(output),
#                            reduction_indices=len(output.get_shape()) - 1)

#def w_categorical_crossentropy(y_true, y_pred, weights):
#    nb_cl = len(weights)
#    final_mask = K.zeros_like(y_pred[:, 0])
#    y_pred_max = K.max(y_pred, axis=1)
#    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
#    y_pred_max_mat = K.equal(y_pred, y_pred_max)
#    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
#    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def w_categorical_accuracy(y_true, y_pred, weights):
    '''Calculates the mean accuracy rate across all predictions for
    multiclass classification problems.
    '''
    
    return K.mean(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(tf.multiply(K.variable(weights), y_pred), axis=-1)))

w_array = np.ones(nb_classes)
no_logo_index = valid_generator.class_indices['no-logo']
w_array[no_logo_index] = 75.0
w_array = w_array / np.linalg.norm(w_array)
pprint.pprint(w_array)


ncce = partial(w_categorical_accuracy, weights=w_array)
ncce.__name__ ='w_categorical_accuracy'

model.compile(loss=ncce,
        optimizer=Adam(),
        metrics=[ncce])

if not WRITE_SUBMISSION:
    res = model.evaluate_generator(valid_generator, nb_samples)
    pprint.pprint(res)

if WRITE_SUBMISSION:
    test_generator = test_datagen.flow_from_directory(
            TEST_FOLDER,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=1,
            shuffle=False,
            color_mode=COLOR_MODE,
            class_mode='categorical')
            #save_to_dir='debug_validation')


    submission_file = open('submission.txt', 'w')
    submission_file.write('Id,Prediction\n')

    file_index = 0
    for X_batch, Y_batch in test_generator:
        pred = (model.predict(X_batch, batch_size=1))
        pred = pred * w_array
        #pprint.pprint(inv_map_list)
        index = np.argmax(pred)
        if PLOT_ENABLED and index != no_logo_index:
            print(index)
            print(inv_map[index])
            pprint.pprint(pred)
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(2, 1, 1)
            width = 5
            ax.bar(list(range(nb_classes)), pred[0], 1.0)
            ax.set_xticks([x + 0.5 for x in range(nb_classes)])
            ax.set_xticklabels(inv_map_list, size=7)
            ax = fig.add_subplot(2, 1, 2)
            ax.imshow(X_batch[0])
        filename = os.path.basename(test_generator.filenames[file_index])
        submission_file.write(filename + ',' + inv_map[index] + '\n')
        plt.show()
        file_index += 1
        if file_index >= len(test_generator.filenames):
            break

    submission_file.close()

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


