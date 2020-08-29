import os
from PIL import Image
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from skimage import transform

IMG_SIZE = 24

def load_image(filename):
	np_image = Image.open(filename)
	np_image = np.array(np_image).astype('float32')/255
	np_image = transform.resize(np_image, (IMG_SIZE, IMG_SIZE, 1))
	np_image = np.expand_dims(np_image, axis=0)
	return np_image

def load_data():
	train_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			horizontal_flip=True, 
		)
	val_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			horizontal_flip=True,		)
	train_generator = train_datagen.flow_from_directory(
	    directory="dataset/train",
	    target_size=(IMG_SIZE, IMG_SIZE),
	    color_mode="grayscale",
	    batch_size=32,
	    class_mode="binary",
	    shuffle=True,
	    seed=42
	)
	val_generator = val_datagen.flow_from_directory(
	    directory="dataset/val",
	    target_size=(IMG_SIZE, IMG_SIZE),
	    color_mode="grayscale",
	    batch_size=32,
	    class_mode="binary",
	    shuffle=True,
	    seed=42
	)
	return train_generator, val_generator

def save_model(model):
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model.h5")


