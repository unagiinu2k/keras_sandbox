import os
base_image_name = "cat"
base_image_path = os.path.join("data", base_image_name +".png")

#Keras
from keras.preprocessing.image import load_img, img_to_array
img = load_img(base_image_path)


import PIL