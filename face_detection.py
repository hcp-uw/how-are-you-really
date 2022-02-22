#!/usr/bin/env python3

import dlib
import numpy as np
import urllib
import cv2
import matplotlib.pyplot as plt

frontalface_detector = dlib.get_frontal_face_detector()

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def detect_face(image_url):
    try:
        url_response = urllib.request.urlopen(image_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)

    except Exception:
        print ("Please check the URL and try again!")
        return None

    rects = frontalface_detector(image, 1)
    if len(rects) < 1:
        return "No Face Detected"
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(image, interpolation='nearest')
    plt.axis('off')
    plt.show()