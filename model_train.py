import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation , Dense , Dropout , Flatten
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
import numpy as np
from keras.utils import np_utils
from keras import optimizers
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from utils import get_data , get_no_of_classes , get_image_size
import matplotlib.pyplot as plt
import math
import h5py as h5py

import os
K.set_image_dim_ordering('tf')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
ap = argparse.ArgumentParser()
ap.add_argument( "-e","--epochs", required=True, help="enter the no of epochs")

args = vars(ap.parse_args())

dataset_path = os.path.dirname(os.path.abspath(__file__))+'/data'
model_path = os.path.dirname(os.path.abspath(__file__))+'/model/keras_model.h5'
weight_path = os.path.dirname(os.path.abspath(__file__))+'/model/keras_weight.h5'

input_size = get_image_size(dataset_path)
no_classes = get_no_of_classes(dataset_path)

epochs = int(args["epochs"])
batch_size = 32
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2


def keras_model():
	model = Sequential()
	model.add(Conv2D(32, (5,5), input_shape=(input_size[0], input_size[1], 1), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(10, 10), strides=(10, 10), padding='same'))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.6))
	model.add(Dense(no_classes, activation='softmax'))
	sgd = optimizers.SGD(lr=1e-2)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	checkpoint1 = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]
	return model, callbacks_list

def train():
    train_images,train_labels,test_images,test_labels,val_images,val_labels = get_data()

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    val_images = np.array(val_images)

    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)
    val_labels = np_utils.to_categorical(val_labels)

    model,callbacks_list  = keras_model()

    history = model.fit(train_images,train_labels,validation_data=(val_images,val_labels),epochs=epochs,batch_size=batch_size,callbacks=callbacks_list)


    (eval_loss, eval_accuracy) = model.evaluate(test_images,test_labels,batch_size=batch_size,verbose=0)

    print("accuracy : {:.2f}%".format(eval_accuracy * 100))
    print("loss : {}".format(eval_loss))

    model.save(model_path)
    model.save_weights(weight_path)
    model = load_model(model_path)


    plt.figure(1)
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'lower left')
    plt.savefig('epoch_'+str(epochs)+'_graph.png')
    plt.show()

train()
