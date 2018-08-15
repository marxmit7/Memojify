from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation,Dense,Dropout,Flatten
from keras import backend as K
import numpy as np
import os
from keras import optimizers
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import load_model
from utils import get_data


dataset_path = os.path.dirname(os.path.abspath(__file__))+'/data'


def cnn_model():
    no_classes = len(os.listdir(dataset_path+'/'))
    print(no_classes)
