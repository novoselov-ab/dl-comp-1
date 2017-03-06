'''Deep Dreaming in Keras.

Run the script with:
```
python deep_dream.py path_to_your_base_image.jpg prefix_for_results
```
e.g.:
```
python deep_dream.py img/mypic.jpg results/dream
```

It is preferable to run this script on GPU, for speed.
If running on CPU, prefer the TensorFlow backend (much faster).

Example results: http://i.imgur.com/FX6ROg9.jpg
'''
from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model, Model
import numpy as np
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import pprint
import resnet
import os

from keras.applications import vgg16
from keras import backend as K
from keras.layers import Input

#parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
#parser.add_argument('base_image_path', metavar='base', type=str,
#                    help='Path to the image to transform.')
#parser.add_argument('result_prefix', metavar='res_prefix', type=str,
#                    help='Prefix for the saved results.')
#
#args = parser.parse_args()
#base_image_path = args.base_image_path
result_prefix = 'res_'# args.result_prefix

# dimensions of the generated picture.
img_width = 224 * 4
img_height = 224 * 4

# some settings we found interesting
saved_settings = {
    'bad_trip': {'features': {'convolution2d_15': 0.07},
                 'continuity': 15.0,
                 'dream_l2': 0.2,
                 'jitter': 0.0},
    'bad_trip2': {'features': {'convolution2d_31': 0.001, 'convolution2d_32': 0.002, 'convolution2d_33': 0.003},
                 'continuity': 0.02,
                 'dream_l2': 0.05,
                 'jitter': 0.0},
    'fire': {'features': {'fire4_expand2': 0.02},
                 'continuity': 0.1,
                 'dream_l2': 0.2,
                 'jitter': 0.00},
}
# the settings we will use in this experiment
settings = saved_settings['bad_trip']

IMAGE_DIR = 'imgs'

# util function to open, resize and format pictures into appropriate tensors


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img /= 255.
    img = np.expand_dims(img, axis=0)
    return img

# util function to convert a tensor into a valid image


def deprocess_image(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_width, img_height))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_width, img_height, 3))
    # Remove zero-center by mean pixel
    #x[:, :, 0] += 103.939
    #x[:, :, 1] += 116.779
    #x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    #x = x[:, :, ::-1]
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

if K.image_dim_ordering() == 'th':
    img_size = (3, img_width, img_height)
else:
    img_size = (img_width, img_height, 3)
# this will contain our generated image
#dream = Input(batch_shape=(1,) + img_size)

#model = load_model('../submitted_models/attempt_Epy20170118-002137[RMSprop_lr0005_d0_l20002_d5]69-0.73.hdf5')
model = resnet.ResnetBuilder.build_resnet_50((3, img_width, img_height), 33)
model.load_weights('../saved_good_models/attempt_resnet_2py20170125-193142[ResNet50]RMSprop_lr0.0002_d0.0_l20.00500_d0.5_2_14-0.89.hdf5')
model.summary()
dream = model.input
print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# continuity loss util function


def continuity_loss(x):
    assert K.ndim(x) == 4
    if K.image_dim_ordering() == 'th':
        a = K.square(x[:, :, :img_width - 1, :img_height - 1] -
                     x[:, :, 1:, :img_height - 1])
        b = K.square(x[:, :, :img_width - 1, :img_height - 1] -
                     x[:, :, :img_width - 1, 1:])
    else:
        a = K.square(x[:, :img_width - 1, :img_height - 1, :] -
                     x[:, 1:, :img_height - 1, :])
        b = K.square(x[:, :img_width - 1, :img_height - 1, :] -
                     x[:, :img_width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# define the loss
loss = K.variable(0.)
for layer_name in settings['features']:
    # add the L2 norm of the features of a layer to the loss
    assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
    coeff = settings['features'][layer_name]
    x = layer_dict[layer_name].output
    shape = layer_dict[layer_name].output_shape
    # we avoid border artifacts by only involving non-border pixels in the loss
    if K.image_dim_ordering() == 'th':
        loss -= coeff * K.sum(K.square(x[:, :, 2: shape[2] - 2, 2: shape[3] - 2])) / np.prod(shape[1:])
    else:
        loss -= coeff * K.sum(K.square(x[:, 2: shape[1] - 2, 2: shape[2] - 2, 1:5])) / np.prod(shape[1:])

# add continuity loss (gives image local coherence, can result in an artful blur)
loss += settings['continuity'] * continuity_loss(dream) / np.prod(img_size)
# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
loss += settings['dream_l2'] * K.sum(K.square(dream)) / np.prod(img_size)

# feel free to further modify the loss as you see fit, to achieve new effects...

# compute the gradients of the dream wrt the loss
grads = K.gradients(loss, dream)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([dream, K.learning_phase()], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1,) + img_size)
    outs = f_outputs([x, 0])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the loss

for filepath in os.listdir(IMAGE_DIR):
    image_path = os.path.join(IMAGE_DIR, filepath)

    x = preprocess_image(image_path)
    for i in range(50):
        print('Start of iteration', i)
        start_time = time.time()

        # add a random jitter to the initial image. This will be reverted at decoding time
        random_jitter = (settings['jitter'] * 2) * (np.random.random(img_size) - 0.5)
        x += random_jitter

        # run L-BFGS for 7 steps
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                        fprime=evaluator.grads, maxfun=7)
        print('Current loss value:', min_val)
        # decode the dream and save it
        x = x.reshape(img_size)
        x -= random_jitter
        img = deprocess_image(np.copy(x))
        fname = 'deep_output/' + str(time.strftime('%Y%m%d-%H%M%S')) + '_' + filepath + result_prefix + '_at_iteration_%d.png' % i
        imsave(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
