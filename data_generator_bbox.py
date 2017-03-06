from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, Iterator, random_channel_shift
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pprint
import sys
import time
import os
import shutil
import utils
import pickle
import random
from PIL import Image as pil_image

IMAGE_SIZE = 299
BATCH_SIZE = 30
BLIT_FRACTION = 1.0
SOURCE_DATA_FOLDER = 'processed_train_masks_0_4cuts'
RGB = True
COLOR_MODE = 'rgb' if RGB else 'grayscale'
NB_SAMPLES_PER_CLASS = 2000
DIST_DATA_FOLDER = 'generated_train_masks_0_4cuts_299x299_' + str(NB_SAMPLES_PER_CLASS)

def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    img = pil_image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]), pil_image.LANCZOS)
    return img

class DirectoryIteratorBBox(Iterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False, bbox_dict=None, blit_mode=False):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.bbox_dict = bbox_dict
        self.blit_mode = blit_mode
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
                    if os.path.splitext(fname)[0].endswith('_mask'):
                        is_valid = False
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
                    if os.path.splitext(fname)[0].endswith('_mask'):
                        is_valid = False
                    if is_valid:
                        self.classes[i] = self.class_indices[subdir]
                        i += 1
                        # add filename relative to directory
                        absolute_path = os.path.join(root, fname)
                        file = os.path.relpath(absolute_path, directory)
                        self.filenames.append(file)

        # collect no-logo hack
        self.no_logo_files = []
        subpath = os.path.join(directory, 'no-logo')
        for root, _, files in _recursive_list(subpath):
            for fname in files:
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if os.path.splitext(fname)[0].endswith('_mask'):
                    is_valid = False
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    file = os.path.relpath(absolute_path, directory)
                    self.no_logo_files.append(file)

        super(DirectoryIteratorBBox, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        batch_bbox = [None] * current_batch_size
        batch_mask = [None] * current_batch_size
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            fbase, fext = os.path.splitext(fname)
            mask_img_path = os.path.join(self.directory, fbase + '_mask' + fext)
            mask_img_size = self.target_size[:2]
            #print(mask_img_size)
            mask_exists = False
            if os.path.exists(mask_img_path):
                mask_img = load_img(mask_img_path, grayscale=True, target_size=mask_img_size)
                mask_img = img_to_array(mask_img, dim_ordering=self.dim_ordering)
                mask_exists = True
            else:
                mask_img = np.zeros(mask_img_size + (1,))
            #np.append(x, mask_img, axis=2)
            csr = self.image_data_generator.channel_shift_range
            x = random_channel_shift(x, csr, self.image_data_generator.channel_axis-1)
            self.image_data_generator.channel_shift_range = 0
            x = np.concatenate((x, mask_img), axis=2)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            self.image_data_generator.channel_shift_range = csr
            #split back
            a =  x[:,:,:3]
            b = x[:,:,3:]
            batch_x[i] = a
            batch_mask[i] = b
                
            batch_bbox[i] = utils.bbox2_unit(b) if mask_exists else (1,0,1,0)
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                if self.blit_mode:
                    no_logo_file = random.choice(self.no_logo_files)
                    no_logo_img = load_img(os.path.join(self.directory, no_logo_file),
                                grayscale=grayscale,
                                target_size=self.target_size)
                    no_logo_img = img_to_array(no_logo_img, dim_ordering=self.dim_ordering)
                    no_logo_img *= 1/255.
                    m = batch_mask[i]
                    no_logo_img = no_logo_img * (1 - m) + batch_x[i] * m
                    img = array_to_img(no_logo_img, self.dim_ordering, scale=True)
                else:
                    img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                
                img.save(os.path.join(self.save_to_dir, fname))
                if self.bbox_dict != None:
                    self.bbox_dict[fname] = batch_bbox[i]
        return batch_x

def generate(source_data_dir, dist_data_dir, augment = True, count = None, bbox_dict = None):
    augment_datagen = ImageDataGenerator(rescale=1./255,
                                #samplewise_center=True,
                                #samplewise_std_normalization=True,
                                #zca_whitening=False,
                                width_shift_range=1.6,
                                height_shift_range=1.6,
                                #channel_shift_range=0.2,
                                rotation_range=30,
                                shear_range=0.3,
                                zoom_range=[2.7, 6.0],
                                channel_shift_range=15.0,
                                fill_mode='constant',
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

        allow_blit = False
        if not augment:
            datagen = dummy_datagen
        elif dirname == 'no-logo':
            datagen = no_logo_datagen
        else:
            datagen = augment_datagen
            allow_blit = True

        train_generator = DirectoryIteratorBBox(source_data_dir, datagen,
                                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),  # all images will be resized to IMAGE_SIZExIMAGE_SIZE
                                                            batch_size=1,
                                                            color_mode=COLOR_MODE,
                                                            class_mode='categorical',
                                                            classes=[dirname],
                                                            save_prefix='YYY',
                                                            save_to_dir=dist_dir,
                                                            bbox_dict=bbox_dict)

        total = len(train_generator.filenames) if count is None else count

        i = 0
        for _ in train_generator:
            i += 1
            if allow_blit:
                if i > total * (1. - BLIT_FRACTION):
                    train_generator.blit_mode = True
            if i >= total:
                break


bbox_dict = {}
generate(SOURCE_DATA_FOLDER + '/train', DIST_DATA_FOLDER + '/train', True, NB_SAMPLES_PER_CLASS, bbox_dict)
generate(SOURCE_DATA_FOLDER + '/validation', DIST_DATA_FOLDER + '/validation', False, None, bbox_dict)

#pprint.pprint(bbox_dict)

if bbox_dict != None:
    pickle.dump(bbox_dict, open(os.path.join(DIST_DATA_FOLDER, 'bbox_dict.pickle'), 'wb'))
