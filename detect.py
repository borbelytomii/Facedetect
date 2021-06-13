import config as url
import cv2
import sys
import logging as log
import datetime as datetime
#from __future__ import print_function
import argparse

url.URL # config.py URl = 'x.x.x.x'


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


capture = cv2.VideoCapture(url.URL)

while(True):
    # Capturing frame by frame
    ret,frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    cv2.imshow('livestream',frame)


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # Exiting the app
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break


# Releasing the VideoCapture
capture.release()
cv2.destroyAllWindows()
