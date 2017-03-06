'''Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes (with the TensorFlow backend).

Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
import math
import os
from keras.models import load_model, Model
from keras.layers import Convolution2D, SeparableConvolution2D
#from keras import backend as K
from keras.backend import tensorflow_backend as K
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import pprint

# dimensions of the generated pictures for each filter.
img_width = 299
img_height = 299
OUTPUT_DIR = 'output'

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def deprocess_images(imgs):
    arr = np.asarray(imgs)
    # normalize tensor: center on 0., ensure std is 0.1
    arr -= arr.mean()
    arr /= (arr.std() + 1e-5)
    arr *= 0.1

    # clip to [0, 1]
    arr += 0.5
    arr = np.clip(arr, 0, 1)

    # convert to RGB array
    arr *= 255
    arr = np.clip(arr, 0, 255).astype('uint8')
    return list(arr)

def stitch_and_save(imgs, name):
    if len(imgs) == 0:
        return

    N = int(math.sqrt(len(imgs))) + 1
    w = imgs[0].shape[0]
    h = imgs[0].shape[1]
    margin = 2
    width = N * w + (N - 1) * margin
    height = N * h + (N - 1) * margin
    stitched_filters = np.zeros((width, height, imgs[0].shape[2]))

    # fill the picture with our saved filters
    for i in range(N):
        for j in range(N):
            num = i * N + j
            if num < len(imgs):
                img = imgs[num]
                stitched_filters[(w + margin) * i: (w + margin) * i + w,
                                (h + margin) * j: (h + margin) * j + h, :] = img

    # save the result to disk
    if stitched_filters.shape[2] == 1:
        stitched_filters = stitched_filters[..., 0]
    
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    imsave('%s/%s_%dx%d_%d.png' % (OUTPUT_DIR, name, w, h, len(imgs)), stitched_filters)

def visualize_layer(model, layer_name):
    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    #test
    #sample_image = load_img('_41_8680.jpeg', grayscale=False)

    kept_filters = []
    layer_output = layer_dict[layer_name].output
    nb_filters = layer_dict[layer_name].output_shape[3]
    for filter_index in range(0, nb_filters):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        #print('Processing filter %d/%d' % (filter_index, nb_filters))
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        if K.image_dim_ordering() == 'th':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])# + tf.multiply(tf.nn.l2_loss(input_img), K.variable(0.0000001))

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img, K.learning_phase()], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_dim_ordering() == 'th':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        #test
        #input_img_data = img_to_array(sample_image)
        #input_img_data = np.expand_dims(input_img_data, axis=0)

        # we run gradient ascent for 20 steps
        for i in range(20):
            print('iterate')
            loss_value, grads_value = iterate([input_img_data, 0])
            input_img_data += grads_value * step

            #print('Current loss value:', loss_value)
            '''if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break'''

        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d/%d (%s) processed in %ds' % (filter_index, nb_filters, layer_name, end_time - start_time))

        # test
        if len(kept_filters) > 15:
            break

    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = [x[0] for x in kept_filters]

    stitch_and_save(kept_filters, 'stitched_filters_%s.png' % (layer_name))


def visualize_layer_weights(model, layer_name):
    print ('visualize_layer_weights: {}'.format(layer_name))

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    weights = layer_dict[layer_name].get_weights()[0]
    weights = weights.reshape((weights.shape[0], weights.shape[1], -1))
    imgs = []
    for ind in range(weights.shape[-1]):
        kernel_3d = weights[..., ind]
        kernel_3d = np.expand_dims(kernel_3d, axis=2)
        img = deprocess_image(kernel_3d)
        imgs.append(img)

    stitch_and_save(imgs, 'layer_weights_{}'.format(layer_name))

def visualize_layer_activations(model, layer_name, filepath):
    intermediate_layer_model = Model(input=model.input,
                                    output=model.get_layer(layer_name).output)

    img = load_img(filepath,
                grayscale=False,
                target_size=None)
    img = img.resize((img_width, img_height))
    img = img_to_array(img)

    X_batch = np.zeros((1,) + img.shape)
    X_batch[0] = img

    intermediate_output = intermediate_layer_model.predict(X_batch, batch_size=1)
    
    output = intermediate_output[0]
    outs = output.reshape((output.shape[0], output.shape[1], -1))

    imgs = []
    for ind in range(outs.shape[-1]):
        feature_activation = outs[..., ind]
        feature_activation = np.expand_dims(feature_activation, axis=2)
        imgs.append(feature_activation)

    imgs = deprocess_images(imgs)

    stitch_and_save(imgs, 'layer_activations_{}_on_{}'.format(layer_name, os.path.basename(filepath)))

    #bonus
    if True:
        prelast_output = Model(input=model.input, output=model.layers[-2].output).predict(X_batch, batch_size=1)
        pprint.pprint(prelast_output)
        last_output = Model(input=model.input, output=model.layers[-1].output).predict(X_batch, batch_size=1)
        pprint.pprint(last_output)


#model = load_model('../saved_models/attempt_Epy20170118-002137[RMSprop_lr0005_d0_l20002_d5]69-0.73.hdf5')
model = load_model('../saved_models/attempt_xception_1py20170203-184436[XCeption]Adam_lr0.001_d0.0_l20.00010_d0.0_2_89-0.92.hdf5')
print('Model loaded.')

model.summary()

VISUALIZE_FEATURES = True
VISUALIZE_WEIGHTS = False
VISUALIZE_OUTPUT = True
INPUT_FILES_FOR_OUTPUT = ['_4_9331.jpeg']

#visualize_layer(model, 'convolution2d_53')
#visualize_layer_activations(model, 'conv10', INPUT_FILES_FOR_OUTPUT[0])
#
#DIR = 'HP'
#for filepath in os.listdir(DIR):
#    visualize_layer_activations(model, 'conv10', os.path.join(DIR, filepath))

for layer in model.layers:
    if(isinstance(layer, Convolution2D) or isinstance(layer, SeparableConvolution2D)):
        pprint.pprint(layer.name)
        pprint.pprint(type(layer))
        if VISUALIZE_FEATURES:
            visualize_layer(model, layer.name)
        if VISUALIZE_WEIGHTS:
            visualize_layer_weights(model, layer.name)
        if VISUALIZE_OUTPUT:
            for filepath in INPUT_FILES_FOR_OUTPUT:
                visualize_layer_activations(model, layer.name, filepath)

#visualize_layer(model, 'convolution2d_11')

#visualize_layer(model, 'convolution2d_11')
#visualize_layer(model, 'convolution2d_12')
#visualize_layer(model, 'convolution2d_13')
#visualize_layer(model, 'convolution2d_14')
#visualize_layer(model, 'convolution2d_15')

