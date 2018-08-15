from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation , Dense , Dropout , Flatten
from keras import backend as K
import numpy as np
import os
from keras import optimizers
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import load_model
from utils import get_data , get_no_of_classes , get_image_size


dataset_path = os.path.dirname(os.path.abspath(__file__))+'/data'



def cnn_model():
    # model = Sequential()
    # model.add(Conv2D(32,(5,5),input_shape=(),activation='relu'))
    # no_classes = no_of_classes(dataset_path)
    # print(no_classes)
    x=get_image_size(dataset_path)
    print(x)

cnn_model()