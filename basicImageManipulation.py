import os
base_image_name = "cat"
base_image_path = os.path.join("data", base_image_name +".png")

#Keras
from keras.preprocessing.image import load_img, img_to_array
imgK = load_img(base_image_path ,target_size=(300,300))

imgK.show()#works fine
type(imgK)
imgK2 = img_to_array(imgK)

type(imgK2)

imgK2.shape
imgK2[2].shape

#pil, pillow
import PIL
imgP = PIL.Image.open(base_image_path , 'r')
type(imgP)
imgP.show()#doesn't work on windows with MS Paint
