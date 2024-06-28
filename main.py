import cv2
import numpy as np
import pytesseract
import os
import time
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

#paths
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\adaryanani\\Desktop\\tesseract-ocr\\tesseract.exe'
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
IMAGE_PATH = "1.png"

def preprocess_image(image_path):
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    #if png, remove alpha channel, reduce down to 3 channels
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    #saves unscaled Tensor Images
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)

def plot_image(image, title=" "):
    #plots images from image tensors
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)


hr_image = preprocess_image(IMAGE_PATH)
model = hub.load(SAVED_MODEL_PATH)
sharpened_image = model(hr_image)
tf.squeeze(sharpened_image)
plot_image(tf.squeeze(sharpened_image), title="sharpened_image")
save_image(tf.squeeze(sharpened_image), filename="sharpened_image")

sharpened = cv2.imread('sharpened_image.jpg')
unsharpened = cv2.imread(IMAGE_PATH)
img = cv2.resize(sharpened, (0,0), fx=0.4, fy=0.4)
'''
cv2.imshow('sharpened', sharpened)
cv2.imshow('unsharpened', unsharpened)
cv2.waitKey(0)
'''

#convert BGR to RGB for cv2
#img = cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)

cong = r'--oem 2 --psm 6'
print(pytesseract.image_to_string(img ,config=cong))

hImg, wImg = img.shape[:2]
boxes = pytesseract.image_to_data(img)
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b = b.split()
        if len(b)==12:
            x,y,w,h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img, (x, y), (w+x, h+y), (0,0,255), 2)




cv2.imshow('result', img)
cv2.waitKey(0)

