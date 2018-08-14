import argparse
import cv2
import numpy as np
import dlib
import os
from imutils import face_utils
from imutils.face_utils import FaceAligner

ap = argparse.ArgumentParser()
ap.add_argument( "-l","--label", required=True, help="label of the image to be captured")
args = vars(ap.parse_args())
label = args["label"]


shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


aligned_face = FaceAligner(shape_predictor, desiredFaceWidth=256)
filepath = os.path.dirname(os.path.abspath(__file__))

frame_counter = 0
img_path = filepath+'/data/'+label

if not os.path.exists(filepath+"/data"):
	os.mkdir(filepath+'/data')
if not os.path.exists(img_path):
	os.mkdir(img_path)


vcam = cv2.VideoCapture(0)
vcam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
vcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

x,y,w,h = 40,50,30,40

while True:
	ret_val , img = vcam.read()
	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = detector(gray_img)

	cv2.rectangle(img, (x, y), (x+w, y+h), (255, 155, 177), 2)
	cv2.imshow('img', img)

	cv2.imshow('gray_img',gray_img)
	keypress = cv2.waitKey(1)
	if keypress%256 == 27:
		print("Escape is pressed, quiting...")
		break
	elif keypress%256 == 32:
		if len(faces)>0:
			face = faces[0]
			shape_68 = shape_predictor(img,face)
			shape = face_utils.shape_to_np(shape_68)




		img_name = "{}.png".format(frame_counter)
		cv2.imwrite(img_path+"/"+ img_name, img)
		print("{} captured ".format(img_name))
		frame_counter += 1

vcam.release()
cv2.destroyAllWindows()