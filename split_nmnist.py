import os
import random
import math
import shutil

train_dir = 'nmnist/notMNIST_large/'

processed_dir = 'processed_nmnist_large'

if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)

os.mkdir(processed_dir)

for dirname in os.listdir(train_dir):
    from_dir = os.path.join(train_dir, dirname)
    
    to_dir_val = os.path.join(processed_dir, 'validation', dirname)
    os.makedirs(to_dir_val)

    to_dir_train = os.path.join(processed_dir, 'train', dirname)
    os.makedirs(to_dir_train)

    files = os.listdir(from_dir)
    random.shuffle(files)

    pivot = 1000
    pivot2 = 10000 + 1000

    for file in files[:pivot]:
        shutil.copyfile(os.path.join(from_dir, file), os.path.join(to_dir_val, file))
    for file in files[pivot:pivot2]:
        shutil.copyfile(os.path.join(from_dir, file), os.path.join(to_dir_train, file))
        

