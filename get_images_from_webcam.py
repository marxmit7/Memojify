import argparse
import cv2
import numpy as np
import dlib
import os
import imutils
from imutils import face_utils
from imutils.face_utils import FaceAligner

filepath = os.path.dirname(os.path.abspath(__file__))


ap = argparse.ArgumentParser()
ap.add_argument( "-l","--label", required=True, help="label of the image to be captured")
args = vars(ap.parse_args())
label = args["label"]


shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


fa = FaceAligner(shape_predictor, desiredFaceWidth=256)

img_path = filepath+'/data/'+label


if not os.path.exists(filepath+"/data"):
	os.mkdir(filepath+'/data')
if not os.path.exists(img_path):
	os.mkdir(img_path)

frame_counter = len(os.listdir(img_path))


vcam = cv2.VideoCapture(0)
vcam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
vcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

x,y,w,h = 40,50,30,40

while True:
	ret_val , img = vcam.read()
	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray_faces = detector(gray_img)

	face = gray_faces[0]
	shape_68 = shape_predictor(img,face)
	shape = face_utils.shape_to_np(shape_68)
	(x,y,w,h) = face_utils.rect_to_bb(face)
	clone = img.copy()
	cv2.rectangle(clone, (x-15, y-20), (x+w+20, y+h+10), (255, 0, 0), 1)
	only_face = imutils.resize(img[y-20:y+h+10,x-15:x+w+20],width=150)

	faceAligned = fa.align(img, gray_img, face)

	cv2.imshow('aligned',faceAligned)
	cv2.imshow('img', clone)
	keypress = cv2.waitKey(1)

	if keypress%256 == 27:
		print("Escape is pressed, quiting...")
		break
	elif keypress%256 == 32:
		img_name = "{}.png".format(frame_counter)
		cv2.imwrite(img_path+"/"+ img_name, faceAligned)
		print("{} captured ".format(img_name))
		frame_counter += 1

vcam.release()
cv2.destroyAllWindows()