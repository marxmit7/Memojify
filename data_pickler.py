import os
import pickle
import cv2
import random
from sklearn.utils import shuffle
import numpy as np


def images_labels(data_path):
    images =[]
    labels = []
    for folder in os.listdir(data_path):
        if not folder.startswith('.'):
            file_path = data_path+'/'+folder
            for image in os.listdir(file_path):
                img = cv2.imread(file_path+'/'+image,0)
                images.append(np.array(img,dtype = np.float16))
                labels.append(str(folder))
    return images,labels

dataset_path = os.path.dirname(os.path.abspath(__file__))+'/data'

images ,labels = images_labels(dataset_path)
print(len(images),len(labels))

for image,label in zip(images,labels):
    print(image,label,"\n")



