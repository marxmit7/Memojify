import cv2
from keras.models import load_model
import numpy as np
import dlib
import os
from imutils.face_utils import FaceAligner
from imutils import face_utils
from utils import get_emojis , get_image_size , get_no_of_classes
from utils import keras_predict , blend

filepath = os.path.dirname(os.path.abspath(__file__))

vcam = cv2.VideoCapture(0)
vcam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
vcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(filepath+"/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(shape_predictor, desiredFaceWidth=256)

# emoji_model = filepath + "/model/keras_model.h5"
# model = load_model(emoji_model)

input_size = get_image_size(filepath+'/data')


def live_feed():
    emojis = get_emojis()

    while True:
        img = vcam.read()[1]
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face = gray_faces[1]

        shape_68 = shape_predictor(img,face)
        shape = face_utils.shape_to_np(shape_68)
        (x,y,w,h) = face_utils.rect_to_bb(face)
        faceAligned = fa.align(img, gray_img, face)

    	cv2.imshow('aligned',faceAligned)
        cv2.imshow('face ', img[y:y+h, x:x+w])
        pred_probab , pred_class = keras_predict(emoji_model, faceAligned)
        img = blend(img, emojis[pred_class], (x, y, w, h))

        cv2.imshow('img', img)
        keypress = cv2.waitKey(1)
        if keypress%256 == 27:
            print("Escape is pressed, quiting...")
            break


live_feed()