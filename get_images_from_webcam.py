import cv2
import numpy as np
import dlib
import os
from imutils import face_utils
from imutils.face_utils import FaceAligner
from random import shuffle, randint

shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

vcam = cv2.VideoCapture(0)
vcam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
vcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

aligned_face = FaceAligner(shape_predictor, desiredFaceWidth=256)
filepath = os.path.dirname(os.path.abspath(__file__))

frame_counter = 0
label = input("enter the label of the image to be captured\n")

if not os.path.exists(filepath+"/data"):
        os.mkdir(filepath+'/data')
if not os.path.exists(filepath+"/data/"+label):
        os.mkdir(filepath+'/data/'+label)


