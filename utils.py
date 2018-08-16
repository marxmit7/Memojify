import os
import cv2
import random
from sklearn.utils import shuffle
import numpy as np

dataset_path = os.path.dirname(os.path.abspath(__file__))+'/data'

if not os.path.exists("model"):
    os.mkdir("model")

def images_labels(data_path):
    image_label=[]
    print("preparing all images as numpy arrays ...\n")
    category = 0
    for folder in os.listdir(data_path):
        if not folder.startswith('.'):
            file_path = data_path+'/'+folder

            for image in os.listdir(file_path):
                img = cv2.imread(file_path+'/'+image,0)
                image_label.append((np.array(img,dtype = np.float16),int(category)))

            shuffle_me= shuffle(shuffle(shuffle(image_label)))
            category +=1

    return shuffle_me


def img_lab(data_array):
    images=[]
    labels =[]
    for (image,label) in data_array:
        image = image.reshape((image.shape[0],image.shape[1],1))
        images.append(image)
        labels.append(label)
    return images,labels


def get_data():

    prepared_data = images_labels(dataset_path)

    training_data = prepared_data[:int(4/5*len(prepared_data))]
    testing_data =  prepared_data[int(4/5*len(prepared_data)):int(9/10*len(prepared_data))]
    validation_data = prepared_data[int(9/10*len(prepared_data)):]

    train_images , train_labels = img_lab(training_data)
    test_images , test_labels = img_lab(testing_data)
    val_images , val_labels = img_lab(validation_data)

    print(" total image available     ",len(prepared_data),"\n no of training images:    ",len(train_images),"\n no of test images:        ",len(test_images),"\n no of validation images:  ",len(val_images))

    return train_images,train_labels,test_images,test_labels,val_images,val_labels


def get_no_of_classes(dataset_path):
    data_dir =[]
    for folder in os.listdir(dataset_path):
        if not folder.startswith('.'):
            data_dir.append(folder)
    return int(len(data_dir))

def get_image_size(dataset_path):
	img = cv2.imread(dataset_path+'/smile/0.png', 0)
	return img.shape


def get_emojis():
    emojis_folder = os.path.dirname(os.path.abspath(__file__))+'/emojis/'
    emojis =[]
    for emoji in os.listdir(emojis_folder):
        if not emoji.startswith('.'):
            emojis.append(cv2.imread(emojis_folder+emoji,-1))
    return emojis


def blend_transparent(face_img, overlay_t_img):
    overlay_img = overlay_t_img[:,:,:3]
    overlay_mask = overlay_t_img[:,:,3:]
    background_mask = 255 - overlay_mask
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

def blend(image, emoji, position):
	x, y, w, h = position
	emoji = cv2.resize(emoji, (w, h))
	try:
		image[y:y+h, x:x+w] = blend_transparent(image[y:y+h, x:x+w], emoji)
	except:
		pass
	return image


def keras_process_image(img):
	input_size = get_image_size(dataset_path)

	img = cv2.resize(img,( input_size[0],input_size[1]))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (-1, input_size[0],input_size[1], 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred = model.predict(processed)
	pred_probab = pred[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class
