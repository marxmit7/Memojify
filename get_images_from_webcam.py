import cv2
import numpy as np
import dlib
import os
from imutils import face_utils
from imutils.face_utils import FaceAligner
from random import shuffle, randint

shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


aligned_face = FaceAligner(shape_predictor, desiredFaceWidth=256)
filepath = os.path.dirname(os.path.abspath(__file__))

frame_counter = 0
label = input("enter the label of the image to be captured\n")
img_path = filepath+'/data/'+label

if not os.path.exists(filepath+"/data"):
	os.mkdir(filepath+'/data')
if not os.path.exists(img_path):
	os.mkdir(img_path)


vcam = cv2.VideoCapture(0)
vcam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
vcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)


while True:
	ret_val , img = vcam.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = detector(gray)


	cv2.imshow('img', img)

	keypress = cv2.waitKey(1)
	if keypress%256 == 27:
		print("escape is pressed quiting...")
		break
	elif keypress%256 == 32:
		img_name = "{}.png".format(frame_counter)
		cv2.imwrite(img_path+"/"+ img_name, img)
		print("{} captured ".format(img_name))
		frame_counter += 1

vcam.release()
cv2.destroyAllWindows()