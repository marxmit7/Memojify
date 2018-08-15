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
from keras.callbacks import TensorBoard , ModelCheckpoint
from keras.models import load_model
from utils import get_data , get_no_of_classes , get_image_size


dataset_path = os.path.dirname(os.path.abspath(__file__))+'/data'

model_path = os.path.dirname(os.path.abspath(__file__))+'/model'

input_size = get_image_size(dataset_path)



def cnn_model():
    no_classes = get_no_of_classes(dataset_path)
    model = Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(input_size[0],input_size[1],1),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(no_classes,activation = 'softmax'))
    sgd = optimizers.SGD(lr = 1e-2)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    model_path = model_path +'/model_generated.h5'
    check_point = ModelCheckPoint(model_path,monitor = 'val_acc',verbose =1 ,save_best_only=True,mode = 'max')
    callbacks_list = [check_point]

    return model,callbacks_list

def train():
    train_images,train_labels,test_images,test_labels,val_images,val_labels = get_data()
    print("train function, type of data",type(train_images))


    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)
    val_labels = np_utils.to_categorical(val_labels)

    model ,callbacks_list = cnn_model()
    model.fit(train_images,train_labels,validation_data=(test_images,test_labels),epochs=20,batch_size=100,callbacks=callbacks_list)
    model = load_model('/model/model_generated.h5')
    scores = model.evaluate(val_images,val_labels,verbose=1)
    print("cnn error %0.2f"%(100-scores[1]*100))
    print(len(train_images),len(test_images))



    # no_classes = no_of_classes(dataset_path)
    # print(no_classes)
    # x=get_image_size(dataset_path)
    # print(x)


train()
