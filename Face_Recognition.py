#Import Packages
import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from os import listdir

import gc
gc.collect()
#-----------------------


#For Linux write the same path as i wrote for windows it is different just locate the frontalface_default.xml file in haarcascade in anaconda
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def VggFaceModel():
    #Create All the Layers in VGG-Face model
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))
	
	model.load_weights('vgg_face_weights.h5')
	
	vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
	
	return vgg_face_descriptor

model = VggFaceModel()

#------------------------

#put your pictures in this path as name.jpg
pictures =  "./dataset/"

pic = dict()

for file in listdir(pictures):
	name, extension = file.split(".")
	pic[name] = model.predict(preprocess_image('./dataset/%s.jpg' % (name)))[0,:]

	
print("employee representations retrieved successfully")

#Function to find Cosine Similarity
def findCosineSimilarity(source, test):
    a = np.matmul(np.transpose(source), test)
    b = np.sum(np.multiply(source, source))
    c = np.sum(np.multiply(test, test))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

#------------------------

cap = cv2.VideoCapture(0) #To use webcam 
#cap = cv2.VideoCapture('/video_name.mp4') #video

while(True):
	retval, img = cap.read()

	if retval:
		img = cv2.resize(img, (1000, 600))

		# haarclassifiers work better in black and white
		gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) #draw rectangle around Detected Face
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 255
			
			captured_representation = model.predict(img_pixels)[0,:]
			
			found = 0
			for i in pic:
				name = i
				representation = pic[i]
				
				similarity = findCosineSimilarity(representation, captured_representation)
				if(similarity < 0.30): # If Cosine Similarity is less than 0.30 then They are matched
					y = y - 6 if y - 6 > 6 else y + 17
					cv2.putText(img, name, (x+4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
					
					found = 1
					break
		
			if(found == 0): #if image is not in database
				cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2) #draw rectangle around Detected Face
				y = y - 6 if y - 6 > 6 else y + 17
				cv2.putText(image, 'DESCONOCIDO', (x+4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
	
		cv2.imshow('Reconocimiento Facial',img)
	
		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break
	
#release Capture and Destroy  WIndows
cap.release()
cv2.destroyAllWindows()
