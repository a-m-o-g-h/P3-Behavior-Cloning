import os
import csv
import numpy as np
import cv2
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda ,PReLU
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint

#code for flipping of images
def flip(images,measurements,prob = 1):
	augmented_images, augmented_measurements =[], []
	for image,measurement in zip(images,measurements):
		augmented_images.append(image)
		augmented_measurements.append(measurement)
		if(np.random.rand() < prob):
			augmented_images.append(cv2.flip(image,1))
			augmented_measurements.append(measurement * -1.0)
	return augmented_images, augmented_measurements

# random brightness, shadow, shift horizon
def random_distort(img, angle):
	new_img = img.astype(float)

	value = np.random.randint(-28, 28)
	if value > 0:
		mask = (new_img[:,:,0] + value) > 255 
	if value <= 0:
		mask = (new_img[:,:,0] + value) < 0
	new_img[:,:,0] += np.where(mask, 0, value)

	h,w = new_img.shape[0:2]
	mid = np.random.randint(0,w)
	factor = np.random.uniform(0.6,0.8)
	if np.random.rand() > .5:
		new_img[:,0:mid,0] *= factor
	else:
		new_img[:,mid:w,0] *= factor

	h,w,_ = new_img.shape
	horizon = 2*h/5
	v_shift = np.random.randint(-h/8,h/8)
	pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
	pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
	return (new_img.astype(np.uint8), angle)	

#batch samples generator	
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: 
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):

			batch_samples = samples[offset:offset+batch_size]
			images = []
			steerings = []

			for batch_sample in batch_samples:
				#choosing approx 25% of near zero steering angle samples
				path = './training_data/IMG/'
				if(-0.02 < float(batch_sample[3]) < 0.02):
					if(np.random.rand() > 0.25):
						continue
				name_center = path + batch_sample[0].split('/')[-1]
				image_center = cv2.imread(name_center)
				name_left = path + batch_sample[1].split('/')[-1]
				image_left = cv2.imread(name_left)
				name_right = path + batch_sample[2].split('/')[-1]
				image_right = cv2.imread(name_right)

				steering_center = float(batch_sample[3])		
				correction = 0.2	#correction of steering value for left and right images
				steering_left = steering_center + correction
				steering_right = steering_center - correction
				images.extend([image_center,image_left,image_right])
				steerings.extend([steering_center,steering_left,steering_right])
				
				img_c,str_c = random_distort(image_center,steering_center)
				images.append(img_c)
				steerings.append(str_c) 

		
			images,steerings = flip(images,steerings,0.5)
			X_train = np.array(images)
			y_train = np.array(steerings)
			yield shuffle(X_train, y_train)


samples = []
with open('./training_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
# data splitted for training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)



model = Sequential()
# From top 50 pixel and bottom 20 pixels are removed
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# Image normalizaion
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# Convolution layer
model.add(Convolution2D(24,5,5,subsample=(2,2)))
# Activation Layer - PReLU
model.add(PReLU())

model.add(Convolution2D(36,5,5,subsample=(2,2)))

model.add(PReLU())

model.add(Convolution2D(48,5,5,subsample=(2,2)))

model.add(PReLU())

model.add(Convolution2D(64,3,3))

model.add(PReLU())

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(500))

model.add(PReLU())

model.add(Dense(200))

model.add(PReLU())

model.add(Dense(40))

model.add(PReLU())

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# checkpoint concept was implemented to store model weights after each epoch
filepath="model_final-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath)
callbacks_list = [checkpoint]

history_object = model.fit_generator(train_generator, samples_per_epoch =8 * len(train_samples), validation_data = validation_generator, nb_val_samples =8 * len(validation_samples), nb_epoch=5, callbacks = callbacks_list , verbose = 1)

model.save('model.h5')

