from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import ReduceLROnPlateau, Callback, LambdaCallback, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, activity_l2
import numpy as np
import matplotlib.pyplot as plt
import pprint
import sys
import os
import time
import resnet

NB_CLASSES = 33
IMAGE_SIZE = 224
#BATCH_SIZE = 30
DATA_FOLDER = 'generated_train_cropped_multiple_05_224x224_bbox_blit_5000' #'real_validation_orig' #generated_train_cropped_224x224_multiple'#'real_validation'
RGB = True
COLOR_MODE = 'rgb' if RGB else 'grayscale'
PLOT_BARS = True

model = resnet.ResnetBuilder.build_resnet_50((3, IMAGE_SIZE, IMAGE_SIZE), NB_CLASSES)
model.summary()
model.load_weights('saved_good_models/attempt_resnet_2py20170125-193142[ResNet50]RMSprop_lr0.0002_d0.0_l20.00500_d0.5_2_14-0.89.hdf5')
model_2 = Model(input=model.input, output=model.get_layer('convolution2d_53').output)

def generator_predictor():
    test_datagen = ImageDataGenerator(rescale=1./255,
            #samplewise_center=True,
            #samplewise_std_normalization=True,
            zca_whitening=False)

    test_generator = test_datagen.flow_from_directory(
            DATA_FOLDER + '/validation',
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=1,
            color_mode=COLOR_MODE,
            class_mode='categorical',
            shuffle=False)
            #save_to_dir='debug_validation')

    nb_classes = len(test_generator.class_indices)
    nb_samples = len(test_generator.filenames)
    inv_map = {v: k for k, v in test_generator.class_indices.items()}
    inv_map_list = [inv_map[x] for x in range(nb_classes)]

    #model.compile(loss=ncce,
    #       optimizer=Adam(),
    #        metrics=[ncce])

    #res = model.evaluate_generator(test_generator, nb_samples)
    #pprint.pprint(res)

    correct = 0
    total = 0

    for X_batch, Y_batch in test_generator:
            #pprint.pprint(X_batch)         
            #pprint.pprint(Y_batch)         
            #pprint.pprint(test_generator.class_indices)
            #pprint.pprint(X_batch.shape)
            pred = (model.predict(X_batch, batch_size=1))
            index = np.argmax(pred)
            true_index = np.argmax(Y_batch)

            if index == true_index:
                #print('Got it!')
                correct += 1

            if index != true_index and False:
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(2, 1, 1)
                width = 5
                ax.bar(list(range(nb_classes)), pred[0], 1.0)
                ax.set_xticks([x + 0.5 for x in range(nb_classes)])
                ax.set_xticklabels(inv_map_list, size=7)
                ax = fig.add_subplot(2, 1, 2)
                ax.imshow(X_batch[0])
                print(index)
                print(inv_map[index])
                print(true_index)
                print(inv_map[true_index])
                plt.show()

            total += 1
            if total >= nb_samples:
                break
    print('{0}/{1}'.format(correct, total))            
                

def evaluate_on_file(filepath, class_index, classes):
    nb_classes = len(classes)

    orig = load_img(filepath,
                grayscale=False,
                target_size=None)
    w, h = orig.size
                
    window = [w, h]
    pos = [0, 0]

    fig = plt.figure(figsize=(20, 20)) if PLOT_BARS else None
    plts = 1

    crops = []
    while True:
        crops.append((pos[0], pos[1], pos[0] + window[0], pos[1] + window[1]))
        pos[0] += window[0]
        if pos[0] + window[0] / 2. > w:
            pos[0] = 0
            pos[1] += window[1]
        
        if pos[1] + window[1] / 2. > h:
            pos[0] = 0
            pos[1] = 0
            window[0] /= 2.
            window[1] /= 2.

        if len(crops) > 0:
            break

    X_batch = np.zeros((len(crops),) + (IMAGE_SIZE, IMAGE_SIZE, 3))

    for i, crop in enumerate(crops):
        img = orig.crop(crop)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        x = img_to_array(img)
        x *= 1./255
        X_batch[i] = x

    preds = (model.predict(X_batch, batch_size=X_batch.shape[0]))
    preds_conv = model_2.predict(X_batch, batch_size=X_batch.shape[0])

    votes = np.zeros(nb_classes)
    for i, pred in enumerate(preds):
        votes += pred
        #index = np.argmax(pred)
        #votes[index] = max(np.max(pred), votes[index])
        if fig != None:
            ax = fig.add_subplot(6, 6, plts)
            plts += 1
            ax.imshow(X_batch[i])

            ax = fig.add_subplot(6, 6, plts)
            plts += 1
            width = 5
            ax.bar(list(range(nb_classes)), votes, 1.0)
            ax.set_xticks([x + 0.5 for x in range(nb_classes)])
            ax.set_xticklabels(classes, size=7, rotation=80)

            average = np.sum(preds_conv[i], axis=2)
            #average = average / np.linalg.norm(average)
            average -= np.min(average)
            average /= np.max(average)
            average = average**2.0

            #N = 5
            #top_N = average.ravel()[np.argsort(average.ravel())[::-1][N]]
            #average[average < top_N] = 0.0

            #print(average.shape)
            #ax = fig.add_subplot(6, 6, plts)
            #plts += 1
            #ax.imshow(average)


    #votes[21] /= len(preds)
    index = np.argmax(votes)

    if index != class_index and fig != None:
        print(filepath)
        print(index)
        print(classes[index])
        print(class_index)
        print(classes[class_index])

        print(preds_conv[0].shape)

        plt.show()

    if fig != None:       
        plt.close(fig)

    return index == class_index
    
    

def manual_predictor():
    val_dir = DATA_FOLDER + '/validation'
    true_index = 0
    dirs = os.listdir(val_dir)
    correct = 0
    correct_per_class = np.zeros(len(dirs))
    total = 0
    dirs = sorted(dirs)
    pprint.pprint(dirs)
    for dirname in dirs:
        from_dir = os.path.join(val_dir, dirname)

        files = os.listdir(from_dir)
        for file in files:
            #if file == '4834606338.jpg':
            if evaluate_on_file(os.path.join(from_dir, file), true_index, dirs):
                correct_per_class[true_index] += 1
                correct += 1
            total += 1

        print('Correct for {0}: {1}'.format(dirs[true_index], correct_per_class[true_index]))
        print('{0}/{1}'.format(correct, total))
        true_index += 1

manual_predictor()   
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


