import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('/home/raj/Documents/project/BrainTumor/Braintumor.keras')
image=cv2.imread('BrainTumor/pred/pred0.jpg')
img=Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)
input_img=np.expand_dims(img,axis=0)
result = np.argmax(model.predict(input_img), axis=-1)# result=model.predict_classes(input_img)
if result==0:
    print('no tumor')
else:
    print('tumor found')