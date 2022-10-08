#!/usr/bin/env python3

import dlib
import numpy as np
import urllib
import cv2
import matplotlib.pyplot as plt

frontalface_detector = dlib.get_frontal_face_detector()
landmark_predictor=dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def get_landmarks(image_url):
    try:
        url_response = urllib.request.urlopen(image_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
    
    except Exception:
        print ("Please check the URL and try again!")
        return None,None
    
    faces = frontalface_detector(image, 1)
    if len(faces):
        landmarks = [(p.x, p.y) for p in landmark_predictor(image, faces[0]).parts()]
    
    else:
        return None,None
    
    return image,landmarks

def image_landmarks(image,face_landmarks):
    radius = -1
    circle_thickness = 4
    image_copy = image.copy()
    for (x, y) in face_landmarks:
        cv2.circle(image_copy, (x, y), circle_thickness, (255,0,0), radius)
        plt.imshow(image_copy, interpolation='nearest')
        plt.axis('off')
        plt.show()