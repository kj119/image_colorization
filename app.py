import tensorflow as tf
import keras
import numpy as np
import os
import streamlit as st
import PIL 
from PIL import Image
import cv2
import time
from keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt 

# adds html elements for title and subtitle
st.markdown("<h1 style='text-align:center;'>Image Colorization</h1>",unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Built with Tensorflow2 & Keras</h3>",unsafe_allow_html=True)

# adds instruction text for the user
st.text('2. Click the button below to colorize your selected image.')

# loads grayscale and colored images
gray=np.load('gray_short.npy')
ab=np.load('ab_short.npy')

# creates a sidebar
st.sidebar.title('1. Choose from 250 images')

# creates number selector that allows user to choose a number and have the corresponding image displayed
i=st.sidebar.number_input(label='Enter a value:',min_value=1,value=1,step=1)

# preps grayscale images for the model
def batch_prep(gray_img,batch_size=100):
	img=np.zeros((batch_size,224,224,3))
	for i in range(0,3):
		img[:batch_size,:,:,1]=gray_img[:batch_size]
	return img

img_in=batch_prep(gray,batch_size=250)

# preps colored images
def get_rbg(gray_imgs, ab_imgs, n=10):
  img1 = np.zeros((n, 224, 224, 3))
  img1[:,:,:,0] = gray_imgs[0:n:]
  img1[:,:,:,1:] = ab_imgs[0:n]
  img1 = img1.astype("uint8")
  imgs = []
  for i in range(0, n):
    imgs.append(cv2.cvtColor(img1[i], cv2.COLOR_LAB2RGB))
  imgs = np.array(imgs)
  return imgs

img_out = get_rbg(gray_imgs=gray, ab_imgs=ab, n=250)

# loads custom image
def load_image(image_file):
	img = Image.open(image_file).convert('L')
	size = 224, 224
	img = img.resize((224, 224), Image.ANTIALIAS)
	return img

# displays image that corresponds with the number the user selects
st.sidebar.image(gray[i])
st.sidebar.image(img_out[i])

# adds the colorize button
start_analyze_file=st.button('Colorize')

upload_file = st.file_uploader('Upload Grayscale Image (.JPG, .JPEG, .PNG)', type=["png", "jpg", "jpeg"])
img = ''
if upload_file is not None:
	file_details = {'FileName': upload_file.name, "FileType":upload_file.type,"FileSize":upload_file.size}
	st.write(file_details)
	# shape is (224, 224, 1)
	img = load_image(upload_file)
	img_data = img_to_array(img)
	# reshape to (1, 224, 224)
	img_data = img_data.reshape((1, 224, 224))
	upload_img = batch_prep(img_data, batch_size=1)

if start_analyze_file==True:

	with st.spinner(text='Colorizing...'):
	     time.sleep(1)

	st.cache(allow_output_mutation=True)
	model=tf.keras.models.load_model('modelfinal.h5')
	
	if upload_file is None:
		prediction=model.predict(img_in)
		st.success('Done!')
		st.image(prediction[i].astype('uint8'),clamp=True)
	elif upload_file is not None:
		prediction = model.predict(upload_img)
		st.success('Done!')
		st.image(img, width=224, height=224)
		st.image(prediction[0].astype('uint8'), clamp=True)



      

