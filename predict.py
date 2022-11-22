# make a prediction for a new image.
import numpy as np
from numpy import argmax
from tensorflow.keras.utils import load_img
#from keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model

import glob

from PIL import Image

# load and prepare the image
def load_image(pathname):
	imgs = []
	for file in glob.glob(pathname+'*.png'):
		img = Image.open(file).convert('L')
		img = img.resize((28, 28))
		img = img_to_array(img)
		img = img.reshape(1, 28, 28, 1)
		imgs.append(img)

	imgs = np.vstack(imgs)
	
	# load the image
	#img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	#img = img_to_array(img)
	# reshape into a single sample with 1 channel
	#img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	#img = img.astype('float32')
	imgs = imgs / 255.0
	return imgs

# load an image and predict the class
def run_example():
	# load the image
	imgs = load_image('imgs/')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	predict_value = model.predict(imgs)
	#digit = argmax(predict_value)
	digits = list()
	for i in range(len(predict_value)):
		digits.append(argmax(predict_value[i]))
	
	for i in range(len(digits)):
		print(digits[i], ' : ', predict_value[i, digits[i]])





# entry point, run the example
run_example()