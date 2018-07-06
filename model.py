import numpy
import csv
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Reshape, Flatten, MaxPooling2D, Dropout
from keras.utils import np_utils

with open("data/train.csv") as trainingData:
    trainingData = csv.reader(trainingData)
    xTrain = trainingData.reshape((trainingData[0],1,48,48))

with open("data/testPublic.csv") as testPublicData:
    testPublicData = csv.reader(testPublicData)
    xTest = testPublicData.reshape((testPublicData[0],1,48,48))

print(xTrain.shape)