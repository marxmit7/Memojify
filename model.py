import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation , Dense , Dropout , Flatten
from keras import backend as K
import numpy as np
import os
from keras.utils import np_utils
from keras import optimizers
from keras.layers import BatchNormalization
from keras.callbacks import  ModelCheckpoint
from keras.models import load_model
from utils import get_data , get_no_of_classes , get_image_size
import matplotlib.pyplot as plt
import math
import h5py as h5py


dataset_path = os.path.dirname(os.path.abspath(__file__))+'/data'

model_path = os.path.dirname(os.path.abspath(__file__))+'/model/keras_model.h5'
weight_path = os.path.dirname(os.path.abspath(__file__))+'/model/keras_weight.h5'


input_size = get_image_size(dataset_path)
no_classes = get_no_of_classes(dataset_path)
epochs = 50

batch_size = 32

nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2


def cnn_model():
    model = Sequential()
    model.add(Conv2D(nb_filters1, (conv1_size,conv1_size), input_shape=(input_size[0], input_size[1], 1),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size),strides=(2, 2), padding='same'))

    model.add(Conv2D(nb_filters2, (conv2_size, conv2_size)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(no_classes, activation='softmax'))
    sgd = optimizers.SGD(lr = 1e-2)
    model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])
    return model


def train():
    train_images,train_labels,test_images,test_labels,val_images,val_labels = get_data()

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    val_images = np.array(val_images)

    print("printing the shape: ",train_images[0].shape)

    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)
    val_labels = np_utils.to_categorical(val_labels)


    model  = cnn_model()
    history = model.fit(train_images,train_labels,validation_data=(val_images,val_labels),epochs=epochs,batch_size=batch_size)
    model.save(model_path)
    model.save_weights(weight_path)


    model = load_model(model_path)

    (eval_loss, eval_accuracy) = model.evaluate(test_images,test_labels,batch_size=batch_size,verbose=1)


    print("accuracy : {:.2f}%".format(eval_accuracy * 100))
    print("loss : {}".format(eval_loss))

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
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()
    plt.savefig('generated_graph.png')


train()
