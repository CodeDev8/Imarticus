import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

CATEGORIES = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

model=keras.models.load_model('MobileNetV2TrainedOnBgSubtraction.h5',custom_objects={'KerasLayer': hub.KerasLayer})

st.title("Sign Language Detection")
run = st.checkbox("Turn ON Camera")
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)

while run:
  ret, frame = cam.read()
  mirror = cv2.flip(frame, 1)
  fh, fw = mirror.shape[:2]
  rois = int(fh/1.7)
  cropImg = mirror[0:rois, fw-rois:fw]
  grey = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
  value = (11, 11)
  blurred = cv2.GaussianBlur(grey, value, 0)
  _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  threed = np.repeat(thresh[..., np.newaxis], 3, -1)
  mirror[0:rois, fw-rois:fw] = threed
  rgb = cv2.cvtColor(mirror, cv2.COLOR_BGR2RGB)
  resizedimg = cv2.resize(threed, (224,224), interpolation = cv2.INTER_CUBIC)
  normalizedingformodel = resizedimg/255.0
  predictions = model.predict(np.array([normalizedingformodel])) 

  if predictions.max()>0.7:
    guessNo = np.argmax(np.squeeze(predictions))
    guessAlpha= CATEGORIES[guessNo]
    cv2.putText(rgb, guessAlpha, (50, fh-10), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 7, cv2.LINE_AA)

  FRAME_WINDOW.image(rgb)
else:
  st.write('Camera is OFF. Turn On for ASL Detection')

