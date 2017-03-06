from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import sys
import os
import random
import math
import shutil
import pickle

TRAIN_DIR = 'BLR_dataset/dataset/train/jpg'
MASKS_DIR = 'BLR_dataset/dataset/train/labels'
CROP = True
FOUR_CUTS = True

processed_dir = 'processed_train_masks_0_4cuts'

if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)

os.mkdir(processed_dir)

VALIDATION_FRACTION = 0.0
MAX_CROP_SCALES = 5
INITIAL_EXTEND = 0
SAVE_MASKS = True

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def masked_crop(file, from_dir, mask_dir, to_dir, bbox_dict, max_nb_masks = -1, save_mask=False):
    mask_index = 0
    while True:
        if max_nb_masks > 0 and mask_index >= max_nb_masks:
            break

        mask_file = os.path.join(mask_dir, file + '.mask.' + str(mask_index) + '.png')
        if not os.path.exists(mask_file):
            if mask_index == 0:
                print('[ERROR] no mask:' + mask_file)
            break
        mask_img = load_img(mask_file, grayscale=True)
        x = img_to_array(mask_img)
        (y0,y1,x0,x1) = bbox2(x)

        (bx0, by0) = (x0, y0)
        (bxw, bxh) = (x1 - x0, y1 - y0)

        w, h = mask_img.size
        dx = x1 - x0
        dy = y1 - y0

        dx = max(dx, w / MAX_CROP_SCALES)
        dy = max(dy, h / MAX_CROP_SCALES)

        x0 = max(x0 - INITIAL_EXTEND, 0)
        x1 = min(x1 + INITIAL_EXTEND, w)
        y0 = max(y0 - INITIAL_EXTEND, 0)
        y1 = min(y1 + INITIAL_EXTEND, h)

        scale_index = 0
        while True:
            orig = load_img(os.path.join(from_dir, file), grayscale=False)
            filename = os.path.splitext(file)[0] + '_' + str(mask_index) + '_' + str(scale_index) + '.png'
            img_save_to = os.path.join(to_dir, filename)
            orig.crop((x0, y0, x1, y1)).save(img_save_to)
            if bbox_dict != None:
                crop_w = float(x1 - x0)
                crop_h = float(y1 - y0)
                bbox_dict[filename] = ((bx0 - x0) / crop_w, (by0 - y0) / crop_h, bxw / crop_w, bxh / crop_h)
            if save_mask:
                mask_img_save_to = os.path.join(to_dir, filename[:-4] + '_mask.png')
                mask_img.crop((x0, y0, x1, y1)).save(mask_img_save_to)
            if MAX_CROP_SCALES == 1:
                break
            if x1 - x0 >= w or y1 - y0 >= h:
                break
            x0 = max(x0 - dx / 2, 0)
            x1 = min(x1 + dx / 2, w)
            y0 = max(y0 - dy / 2, 0)
            y1 = min(y1 + dy / 2, h)
            dx *= 1.5
            dy *= 1.5
            scale_index += 1
        mask_index += 1

def masked_crop_4cuts(file, from_dir, mask_dir, to_dir, bbox_dict, max_nb_masks = -1, save_mask=False):
    mask_index = 0
    while True:
        if max_nb_masks > 0 and mask_index >= max_nb_masks:
            break

        mask_file = os.path.join(mask_dir, file + '.mask.' + str(mask_index) + '.png')
        if not os.path.exists(mask_file):
            if mask_index == 0:
                print('[ERROR] no mask:' + mask_file)
            break
        mask_img = load_img(mask_file, grayscale=True)
        x = img_to_array(mask_img)

        for c in range(4):
            (y0,y1,x0,x1) = bbox2(x)

            x0 = (x1 + x0) * 0.5 if c in [1, 2] else x0
            y0 = (y1 + y0) * 0.5 if c in [2, 3] else y0
            x1 = (x1 + x0) * 0.5 if c in [0, 3] else x1
            y1 = (y1 + y0) * 0.5 if c in [0, 1] else y1

            (bx0, by0) = (x0, y0)
            (bxw, bxh) = (x1 - x0, y1 - y0)

            w, h = mask_img.size

            x0 = max(x0 - INITIAL_EXTEND, 0)
            x1 = min(x1 + INITIAL_EXTEND, w)
            y0 = max(y0 - INITIAL_EXTEND, 0)
            y1 = min(y1 + INITIAL_EXTEND, h)

            orig = load_img(os.path.join(from_dir, file), grayscale=False)
            filename = os.path.splitext(file)[0] + '_' + str(mask_index) + '_cut' + str(c) + '.png'
            img_save_to = os.path.join(to_dir, filename)
            orig.crop((x0, y0, x1, y1)).save(img_save_to)
            if bbox_dict != None:
                crop_w = float(x1 - x0)
                crop_h = float(y1 - y0)
                bbox_dict[filename] = ((bx0 - x0) / crop_w, (by0 - y0) / crop_h, bxw / crop_w, bxh / crop_h)
            if save_mask:
                mask_img_save_to = os.path.join(to_dir, filename[:-4] + '_mask.png')
                mask_img.crop((x0, y0, x1, y1)).save(mask_img_save_to)
        mask_index += 1

bbox_dict = {}
for dirname in os.listdir(TRAIN_DIR):
    from_dir = os.path.join(TRAIN_DIR, dirname)
    from_mask_dir = os.path.join(MASKS_DIR, dirname)
    
    to_dir_val = os.path.join(processed_dir, 'validation', dirname)
    os.makedirs(to_dir_val)

    to_dir_train = os.path.join(processed_dir, 'train', dirname)
    os.makedirs(to_dir_train)

    files = os.listdir(from_dir)
    random.shuffle(files)

    pivot = int(math.floor(len(files) * VALIDATION_FRACTION))

    crop_fn = masked_crop_4cuts if FOUR_CUTS else masked_crop

    for file in files[:pivot]:
        if CROP and dirname != 'no-logo':
            crop_fn(file, from_dir, from_mask_dir, to_dir_val, bbox_dict, save_mask=SAVE_MASKS)
        else:
            shutil.copyfile(os.path.join(from_dir, file), os.path.join(to_dir_val, file))
    for file in files[pivot:]:
        if CROP and dirname != 'no-logo':
            crop_fn(file, from_dir, from_mask_dir, to_dir_train, bbox_dict, save_mask=SAVE_MASKS)
        else:
            shutil.copyfile(os.path.join(from_dir, file), os.path.join(to_dir_train, file))


if bbox_dict != None:
    pickle.dump(bbox_dict, open(os.path.join(processed_dir, 'bbox_dict.pickle'), 'wb'))

