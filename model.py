import numpy as np
import os
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Reshape, Flatten, MaxPooling2D, Dropout
from keras.utils import np_utils

xTrainPath = os.getcwd() + "/data/train.csv"
with open(xTrainPath,'r') as trainingData:
    trainingData = pd.read_csv(trainingData)
    # print(trainingData.values.shape)
    # xTrain = trainingData.reshape((trainingData[0],1,48,48))


xtestPublicPath = os.getcwd()+ "/data/testPublic.csv"
with open(xtestPublicPath,'r') as testPublicData:
    testPublicData = pd.read_csv(testPublicData)
    # xTest = testPublicData.reshape((testPublicData[0],1,48,48))

# print(xTrain.shape)

labels = trainingData.values[:,0]
# print(labels.shape)
tem = trainingData.values[:,1]
pixels = np.zeros((trainingData[0],48*48))