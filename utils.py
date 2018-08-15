import os
import cv2
import random
from sklearn.utils import shuffle
import numpy as np

dataset_path = os.path.dirname(os.path.abspath(__file__))+'/data'

def images_labels(data_path):
    images = []
    labels = []
    image_label=[]
    print("preparing all images as numpy arrays ...\n")
    for folder in os.listdir(data_path):
        if not folder.startswith('.'):
            file_path = data_path+'/'+folder

            for image in os.listdir(file_path):
                img = cv2.imread(file_path+'/'+image,0)
                image_label.append((np.array(img,dtype = np.float16),str(folder)))

            shuffle_me= shuffle(shuffle(shuffle(image_label)))

    return shuffle_me


def img_lab(data_array):
    images=[]
    labels =[]
    for (image,label) in data_array:
        images.append(image)
        labels.append(label)
    return images,labels


def get_data(images_labels,img_lab,dataset_path):

    prepared_data = images_labels(dataset_path)

    training_data = prepared_data[:int(4/5*len(prepared_data))]
    testing_data =  prepared_data[int(4/5*len(prepared_data)):int(9/10*len(prepared_data))]
    validation_data = prepared_data[int(9/10*len(prepared_data)):]

    train_images , train_labels = img_lab(training_data)
    test_images , test_labels = img_lab(testing_data)
    val_images , val_labels = img_lab(validation_data)

    print(len(train_images),len(train_labels),len(test_images),len(test_labels),len(val_images),len(val_labels))

    return train_images,train_labels,test_images,test_labels,val_images,val_labels


def get_no_of_classes(dataset_path):
    data_dir =[]
    for folder in os.listdir(dataset_path):
        if not folder.startswith('.'):
            data_dir.append(folder)
    return data_dir

def get_image_size(dataset_path):
	img = cv2.imread(dataset_path+'/smile/0.png', 0)
	return img.shape