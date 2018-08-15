import os
import pickle
import cv2
import random
from sklearn.utils import shuffle
import numpy as np

dataset_path = os.path.dirname(os.path.abspath(__file__))+'/data'

def images_labels(data_path):
    images = []
    labels = []
    image_label=[]

    for folder in os.listdir(data_path):
        if not folder.startswith('.'):
            file_path = data_path+'/'+folder
            for image in os.listdir(file_path):
                img = cv2.imread(file_path+'/'+image,0)
                image_label.append((np.array(img,dtype = np.float16),str(folder)))

            shuffle_me= shuffle(shuffle(shuffle(image_label)))

            for image, label in shuffle_me:
                images.append(image)
                labels.append(label)
        print(len(images),len(labels))

    return images,labels









