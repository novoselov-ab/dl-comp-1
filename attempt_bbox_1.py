import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, Iterator
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam, SGD
from keras.objectives import categorical_crossentropy
from keras.layers import Input, Activation, merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, Callback, LambdaCallback, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2, activity_l2
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pprint
import sys
import time
import os
import pickle

NB_CLASSES = 33
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCH_FILES_FRACTION = 0.25
DATA_FOLDER = 'generated_train_cropped_multiple_2_224x224_bbox_2000'
RGB = True
COLOR_MODE = 'rgb' if RGB else 'grayscale'
ONE_BBOX = False

class DirectoryIteratorBBoxLabels(Iterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False, bbox_dict=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.bbox_dict = bbox_dict
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.nb_sample += 1
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.classes[i] = self.class_indices[subdir]
                        i += 1
                        # add filename relative to directory
                        absolute_path = os.path.join(root, fname)
                        self.filenames.append(os.path.relpath(absolute_path, directory))
        super(DirectoryIteratorBBoxLabels, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        bbox_output_size = 4 if ONE_BBOX else NB_CLASSES * 4
        batch_bbox = np.zeros((current_batch_size,) + (bbox_output_size,))
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            label = self.classes[j]
            invalid_bbox = (1, 0, 1, 0)
            correct_bbox = invalid_bbox
            if self.bbox_dict != None:
                file = os.path.basename(fname)
                if file in self.bbox_dict:
                    correct_bbox = self.bbox_dict[file]
                else:
                    print(file)

                # temp hack
                if label==21:
                    correct_bbox = invalid_bbox
            if ONE_BBOX:
                for p in range(4):
                    batch_bbox[i, p] = correct_bbox[p]
            else:
                for c in range(NB_CLASSES):
                    for p in range(4):
                        if c == label:
                            batch_bbox[i, 4 * c + p] = correct_bbox[p]
                        else:
                            batch_bbox[i, 4 * c + p] = invalid_bbox[p]
            
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        #pprint.pprint('==================')
        #np.set_printoptions(threshold=np.nan)
        #pprint.pprint(batch_y)
        #pprint.pprint(batch_bbox)
        return batch_x, [batch_y, batch_bbox]

def BuildModel(nb_classes, regularizer = None, dropout = 0.5):
    
    inputs=(IMAGE_SIZE, IMAGE_SIZE, 3 if RGB else 1)

    init = 'glorot_uniform'
    input_img = Input(shape=inputs)
    conv1 = Convolution2D(
        96, 7, 7, init=init, W_regularizer=regularizer,
        subsample=(2, 2), border_mode='same', name='conv1', )(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)

    fire2_squeeze = Convolution2D(
        16, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire2_squeeze')(maxpool1)
    fire2_squeeze = BatchNormalization()(fire2_squeeze)
    fire2_squeeze = Activation('relu')(fire2_squeeze)
    fire2_expand1 = Convolution2D(
        64, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire2_expand1')(fire2_squeeze)
    fire2_expand1 = BatchNormalization()(fire2_expand1)
    fire2_expand1 = Activation('relu')(fire2_expand1)
    fire2_expand2 = Convolution2D(
        64, 3, 3, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire2_expand2')(fire2_squeeze)
    fire2_expand2 = BatchNormalization()(fire2_expand2)
    fire2_expand2 = Activation('relu')(fire2_expand2)
    merge2 = merge(
        [fire2_expand1, fire2_expand2], mode='concat', concat_axis=3)

    fire3_squeeze = Convolution2D(
        16, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire3_squeeze')(merge2)
    fire3_squeeze = BatchNormalization()(fire3_squeeze)
    fire3_squeeze = Activation('relu')(fire3_squeeze)
    fire3_expand1 = Convolution2D(
        64, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire3_expand1')(fire3_squeeze)
    fire3_expand1 = BatchNormalization()(fire3_expand1)
    fire3_expand1 = Activation('relu')(fire3_expand1)
    fire3_expand2 = Convolution2D(
        64, 3, 3, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire3_expand2')(fire3_squeeze)
    fire3_expand2 = BatchNormalization()(fire3_expand2)
    fire3_expand2 = Activation('relu')(fire3_expand2)
    merge3 = merge(
        [fire3_expand1, fire3_expand2], mode='concat', concat_axis=3)

    fire4_squeeze = Convolution2D(
        32, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire4_squeeze')(merge3)
    fire4_squeeze = BatchNormalization()(fire4_squeeze)
    fire4_squeeze = Activation('relu')(fire4_squeeze)
    fire4_expand1 = Convolution2D(
        128, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire4_expand1')(fire4_squeeze)
    fire4_expand1 = BatchNormalization()(fire4_expand1)
    fire4_expand1 = Activation('relu')(fire4_expand1)
    fire4_expand2 = Convolution2D(
        128, 3, 3, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire4_expand2')(fire4_squeeze)
    fire4_expand2 = BatchNormalization()(fire4_expand2)
    fire4_expand2 = Activation('relu')(fire4_expand2)
    merge4 = merge(
        [fire4_expand1, fire4_expand2], mode='concat', concat_axis=3)
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4')(merge4)

    fire5_squeeze = Convolution2D(
        32, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire5_squeeze')(maxpool4)
    fire5_squeeze = BatchNormalization()(fire5_squeeze)
    fire5_squeeze = Activation('relu')(fire5_squeeze)
    fire5_expand1 = Convolution2D(
        128, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire5_expand1')(fire5_squeeze)
    fire5_expand1 = BatchNormalization()(fire5_expand1)
    fire5_expand1 = Activation('relu')(fire5_expand1)
    fire5_expand2 = Convolution2D(
        128, 3, 3, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire5_expand2')(fire5_squeeze)
    fire5_expand2 = BatchNormalization()(fire5_expand2)
    fire5_expand2 = Activation('relu')(fire5_expand2)
    merge5 = merge(
        [fire5_expand1, fire5_expand2], mode='concat', concat_axis=3)

    fire6_squeeze = Convolution2D(
        48, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire6_squeeze')(merge5)
    fire6_squeeze = BatchNormalization()(fire6_squeeze)
    fire6_squeeze = Activation('relu')(fire6_squeeze)
    fire6_expand1 = Convolution2D(
        192, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire6_expand1')(fire6_squeeze)
    fire6_expand1 = BatchNormalization()(fire6_expand1)
    fire6_expand1 = Activation('relu')(fire6_expand1)
    fire6_expand2 = Convolution2D(
        192, 3, 3, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire6_expand2')(fire6_squeeze)
    fire6_expand2 = BatchNormalization()(fire6_expand2)
    fire6_expand2 = Activation('relu')(fire6_expand2)
    merge6 = merge(
        [fire6_expand1, fire6_expand2], mode='concat', concat_axis=3)

    fire7_squeeze = Convolution2D(
        48, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire7_squeeze')(merge6)
    fire7_squeeze = BatchNormalization()(fire7_squeeze)
    fire7_squeeze = Activation('relu')(fire7_squeeze)
    fire7_expand1 = Convolution2D(
        192, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire7_expand1')(fire7_squeeze)
    fire7_expand1 = BatchNormalization()(fire7_expand1)
    fire7_expand1 = Activation('relu')(fire7_expand1)
    fire7_expand2 = Convolution2D(
        192, 3, 3, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire7_expand2')(fire7_squeeze)
    fire7_expand2 = BatchNormalization()(fire7_expand2)
    fire7_expand2 = Activation('relu')(fire7_expand2)
    merge7 = merge(
        [fire7_expand1, fire7_expand2], mode='concat', concat_axis=3)

    fire8_squeeze = Convolution2D(
        64, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire8_squeeze')(merge7)
    fire8_squeeze = BatchNormalization()(fire8_squeeze)
    fire8_squeeze = Activation('relu')(fire8_squeeze)
    fire8_expand1 = Convolution2D(
        256, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire8_expand1')(fire8_squeeze)
    fire8_expand1 = BatchNormalization()(fire8_expand1)
    fire8_expand1 = Activation('relu')(fire8_expand1)
    fire8_expand2 = Convolution2D(
        256, 3, 3, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire8_expand2')(fire8_squeeze)
    fire8_expand2 = BatchNormalization()(fire8_expand2)
    fire8_expand2 = Activation('relu')(fire8_expand2)
    merge8 = merge(
        [fire8_expand1, fire8_expand2], mode='concat', concat_axis=3)

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8')(merge8)

    fire9_squeeze = Convolution2D(
        64, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire9_squeeze')(maxpool8)
    fire9_squeeze = BatchNormalization()(fire9_squeeze)
    fire9_squeeze = Activation('relu')(fire9_squeeze)
    fire9_expand1 = Convolution2D(
        256, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire9_expand1')(fire9_squeeze)
    fire9_expand1 = BatchNormalization()(fire9_expand1)
    fire9_expand1 = Activation('relu')(fire9_expand1)
    fire9_expand2 = Convolution2D(
        256, 3, 3, init=init, W_regularizer=regularizer,
        border_mode='same', name='fire9_expand2')(fire9_squeeze)
    fire9_expand2 = BatchNormalization()(fire9_expand2)
    fire9_expand2 = Activation('relu')(fire9_expand2)
    merge9 = merge(
        [fire9_expand1, fire9_expand2], mode='concat', concat_axis=3)

    fire9_dropout = Dropout(dropout, name='fire9_dropout')(merge9)
    conv10 = Convolution2D(
        nb_classes, 1, 1, init=init, W_regularizer=regularizer,
        border_mode='valid', name='conv10')(fire9_dropout)
    # The size should match the output of conv10
    avgpool10 = AveragePooling2D((13, 13), name='avgpool10')(conv10)

    flatten = Flatten(name='flatten')(avgpool10)
    softmax = Activation("softmax", name='softmax')(flatten)

    # bbox connection
    conv_bb_1 = Convolution2D(64, 3, 3, subsample=(2, 2), W_regularizer=regularizer, border_mode='same')(conv10)
    conv_bb_1 = BatchNormalization()(conv_bb_1)
    conv_bb_1 = Activation('relu')(conv_bb_1)
    conv_bb_1 = Convolution2D(128, 3, 3, subsample=(2, 2), W_regularizer=regularizer, border_mode='same')(conv_bb_1)
    conv_bb_1 = BatchNormalization()(conv_bb_1)
    conv_bb_1 = Activation('relu')(conv_bb_1)
    conv_bb_1 = Convolution2D(256, 3, 3, subsample=(2, 2), W_regularizer=regularizer, border_mode='same')(conv_bb_1)
    conv_bb_1 = BatchNormalization()(conv_bb_1)
    conv_bb_1 = Activation('relu')(conv_bb_1)
    flatten_2 = Flatten(name='flatten_2')(conv_bb_1)
    xx = Dense(1024, W_regularizer=regularizer)(flatten_2)
    xx = BatchNormalization()(xx)
    xx = Activation('relu')(xx)
    xx = Dense(256, W_regularizer=regularizer)(xx)
    xx = BatchNormalization()(xx)
    xx = Activation('relu')(xx)
    xx = Dense(4 if ONE_BBOX else NB_CLASSES * 4, W_regularizer=regularizer)(xx)
    xx = Activation('sigmoid', name='bbox_output')(xx)

    return Model(input=input_img, output=[softmax, xx])

def run_experiment(tag, optimizer, regularizer = None, dropout = 0.5):
        params_str = type(optimizer).__name__ + '_lr' + str(K.eval(optimizer.lr)) + '_d' + str(K.eval(optimizer.decay)) + '_l2' + ('%.5f' % regularizer.l2) + '_d' + str(dropout)
        model_name = sys.argv[0].replace('.', '') + str(time.strftime('%Y%m%d-%H%M%S')) + tag + params_str
        print('Fitting Model: ' + model_name)
    
        model = BuildModel(NB_CLASSES, regularizer, dropout)

        model.compile(loss={'softmax' : categorical_crossentropy, 'bbox_output' : 'mean_squared_error'},
                loss_weights={'softmax': 1., 'bbox_output': 1000.0},
                optimizer=optimizer,
                metrics={'softmax' : 'accuracy', 'bbox_output' : 'mean_squared_error'})

        model.summary()

        if False:
            for layer in model.layers:
                pprint.pprint(layer.name)
                pprint.pprint(layer.input_shape)
                pprint.pprint(layer.output_shape)
                pprint.pprint(layer.get_config())
                pprint.pprint(layer.get_weights())

        train_datagen = ImageDataGenerator(rescale=1./255, samplewise_center=False, samplewise_std_normalization=False)
        valid_datagen = ImageDataGenerator(rescale=1./255, samplewise_center=False, samplewise_std_normalization=False)

        bbox_dict = pickle.load(open(os.path.join(DATA_FOLDER, 'bbox_dict.pickle'), 'rb'))        

        train_generator = DirectoryIteratorBBoxLabels(
                DATA_FOLDER + '/train',  # this is the target directory
                train_datagen,
                target_size=(IMAGE_SIZE, IMAGE_SIZE),  # all images will be resized to IMAGE_SIZExIMAGE_SIZE
                batch_size=BATCH_SIZE,
                color_mode=COLOR_MODE,
                class_mode='categorical',
                bbox_dict=bbox_dict)
                #save_to_dir='debug_train')

        # this is a similar generator, for validation data
        validation_generator = DirectoryIteratorBBoxLabels(
                DATA_FOLDER + '/validation',
                valid_datagen,
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE,
                color_mode=COLOR_MODE,
                class_mode='categorical',
                bbox_dict=bbox_dict)
                #save_to_dir='debug_validation')


        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        tb = TensorBoard(log_dir='tb_logs/' + model_name, histogram_freq=0, write_graph=False, write_images=False)
        es = EarlyStopping(patience=30, monitor='val_softmax_acc')
        mc = ModelCheckpoint(filepath='saved_bbox_models/' + model_name + '{epoch:02d}-{val_softmax_acc:.2f}' + '.hdf5', monitor='val_softmax_acc', save_best_only=True, period=1)
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

        class CustomCallback(Callback):
                def on_epoch_end(self, epoch, logs={}):
                    optimizer = self.model.optimizer
                    lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
                    print('\nLR: {:.6f}\n'.format(lr))
                #def on_batch_begin(self, batch, logs={}):
                 #   pprint.pprint(batch)
                        

        custom_cb = CustomCallback()

        samples_count = len(train_generator.filenames)
        pprint.pprint(train_generator.class_indices)

        history = model.fit_generator(
                train_generator,
                samples_per_epoch=int(samples_count * EPOCH_FILES_FRACTION),
                nb_epoch=10000,
                validation_data=validation_generator,
                nb_val_samples=(NB_CLASSES * 10),
                callbacks = [tb, es, mc, custom_cb])
                #callbacks = [es, mc, custom_cb])


#run_experiment('[A2_RMSprop_lr001_l20002]', RMSprop(lr=0.001), l2(0.0002), 1, 1, 0)
#run_experiment('[RMSprop_lr001_l20002]', RMSprop(lr=0.001), l2(0.0002))
#run_experiment('[RMSprop_lr0005_l20002]', RMSprop(lr=0.0005), l2(0.0002))
#run_experiment('[RMSprop_lr001_l20001_A2]', RMSprop(lr=0.001), l2(0.0001))
#run_experiment('[RMSprop_lr001_de6_l20002_A2]', RMSprop(lr=0.001, decay=1e-6), l2(0.0002))
#run_experiment('[RMSprop_lr0005_l20001]', RMSprop(lr=0.0005), l2(0.0001))
#run_experiment('[RMSprop_lr001_l20002]', RMSprop(lr=0.001), l2(0.002), True)
#run_experiment('[RMSprop_lr001_l20002]', RMSprop(lr=0.001), l2(0.00002), True)
#run_experiment('[RMSprop_1]', RMSprop(lr=0.001), l2(0.0002), 0.5)
#run_experiment('[RMSprop_2]', RMSprop(lr=0.001), l2(0.0002), 0.3)
#run_experiment('[RMSprop_3]', RMSprop(lr=0.0005), l2(0.0002), 0.2)
#run_experiment('[RMSprop_4]', RMSprop(lr=0.0005, decay=1e-5), l2(0.0002), 0.1)
run_experiment('[no_tag]', RMSprop(lr=0.001, decay=0), l2(0.005), 0.5)
#run_experiment('[no_tag]', Adam(lr=0.001, decay=1e-5), l2(0.005), 0.5)
#run_experiment('[RMSprop_lr0005_de5_l20002_d7]', RMSprop(lr=0.0005, decay=1e-5), l2(0.0002), 0.7)
#run_experiment('[RMSprop_lr0005_de5_l20002_d5]', RMSprop(lr=0.001, decay=1e-4), l2(0.0002), 0.5)
#run_experiment('[Adam_lr005_l20002]', Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), l2(0.0002), True)
#run_experiment('[SGD_lr01_m9_l20002]', SGD(lr=0.04, decay=0.002, momentum=0.9), l2(0.0002), True)
#run_experiment('[SGD_lr01_m9_l20002]', SGD(lr=0.04, decay=0.0002, momentum=0.9), l2(0.0002), True)
#run_experiment('[SGD_lr01_m9_l20002]', SGD(lr=0.04, decay=0.00002, momentum=0.9), l2(0.0002), True)

#run_experiment('[SGD_lr01_m9_l20002]', SGD(lr=0.001, decay=0.0002, momentum=0.9), None, True)
#un_experiment('[SGD_lr01_m9_l20002]', SGD(lr=0.01, decay=0.00002, momentum=0.9), l2(0.002), True)
#run_experiment('[SGD_lr001_m9_l20002]', SGD(lr=0.001, decay=0.0002, momentum=0.9), l2(0.0002), True)
#run_experiment('[SGD_lr05_m9_l20002]', SGD(lr=0.05, decay=0.0002, momentum=0.9), l2(0.0002), True)
#run_experiment('[RMSprop_lr0001_l20002]', RMSprop(lr=0.0001), l2(0.0002), True)
#run_experiment('[RMSprop_lr00001_l20002]', RMSprop(lr=0.00001), l2(0.0002), True)
#run_experiment('[RMSprop_lr001_de4_l20002_A2]', RMSprop(lr=0.001, decay=1e-4), l2(0.0002))
#run_experiment('[RMSproplr001]', RMSprop(lr=0.001), None, 1, 1, 1)
#run_experiment('[RMSproplr001]', RMSprop(lr=0.001), None, 3, 3, 0)
#run_experiment('[RMSproplr001]', RMSprop(lr=0.001), None, 4, 4, 0)
#run_experiment('[RMSproplr001]', RMSprop(lr=0.001), None, 4, 2, 0)
#for m in range(6):
    #run_experiment('[RMSproplr001][m' + str(m) + ']', RMSprop(lr=0.001), None, m)
#for m in range(3, 6):    
#    run_experiment('[RMSproplr001_l20002][m' + str(m) + ']', RMSprop(lr=0.001), l2(0.0002), m)
#for m in range(6):
#    run_experiment('[RMSproplr0003][m' + str(m) + ']', RMSprop(lr=0.001), None, m)
#for m in range(6):
#    run_experiment('[RMSproplr0001][m' + str(m) + ']', RMSprop(lr=0.001), None, m)
#for m in range(6):
#    run_experiment('[RMSproplr002][m' + str(m) + ']', RMSprop(lr=0.001), None, m)
#run_experiment('[SGD-01-e6-9-T+l205]', SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True))
#run_experiment('[RMSproplr0001]', RMSprop(lr=0.0001))
#run_experiment('[Adamlr01]', Adam(lr=0.01))
#run_experiment('[Adamlr001]', Adam(lr=0.001))
#run_experiment('[Adamlr0001]', Adam(lr=0.0001))
#run_experiment('[RMSproplr01_l1]', RMSprop(lr=0.01), l1())
#run_experiment('[RMSproplr001_l1]', RMSprop(lr=0.001), l1())
#run_experiment('[RMSproplr0001_l1]', RMSprop(lr=0.0001), l1())
#run_experiment('[Adamlr01_l1]', Adam(lr=0.01), l1())
#run_experiment('[Adamlr001_l1]', Adam(lr=0.001), l1())
#run_experiment('[Adamlr0001_l1]', Adam(lr=0.0001), l1())
#run_experiment('[RMSproplr01_l2]', RMSprop(lr=0.01), l2())
#run_experiment('[RMSproplr001_l2]', RMSprop(lr=0.001), l2())
#run_experiment('[RMSproplr0001_l2]', RMSprop(lr=0.0001), l2())
#run_experiment('[Adamlr01_l2]', Adam(lr=0.01), l2())
#run_experiment('[Adamlr001_l2]', Adam(lr=0.001), l2())
#run_experiment('[Adamlr0001_l2]', Adam(lr=0.0001), l2())


