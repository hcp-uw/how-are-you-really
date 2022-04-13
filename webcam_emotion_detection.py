# load saved model
from tensorflow.keras.models import model_from_json
model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')
from PIL import Image as im
import numpy as np
from numpy import asarray

# load Haar-cascade used to detect position of face
import cv2
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#TODO: replace with image array passed in
# Turns array into image
def readFace(imgArray):

    # make prediction
    # haar_cascade only takes grayscale images; convert into grayscale
    gray_image = cv2.cvtColor(imgArray, cv2.COLOR_BGR2GRAY)
    # detect position of face
    faces = face_haar_cascade.detectMultiScale(gray_image)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    try:
        for (x, y, w, h) in faces:
            # draw rectangle around detected face
            cv2.rectangle(imgArray, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
            # tailor image to feed model
            roi_gray = gray_image[y-5:y+h+5, x-5:x+w+5]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = asarray(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0) # convert 3d matrix into 4d tensor
            print(len(image_pixels[0]))
            print(image_pixels[0])
            #image_pixels /= 255 # normalize
            temp = image_pixels / 255
            # use model to predict emotion
            predictions = model.predict(temp)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            # set overlay
            # TODO: customize
            return {"Emotion": emotion_prediction,
                "Confidence": str(np.round(np.max(predictions[0]) * 100, 1)) + "%"}
    except:
        pass

    # display modified frame
    # type q to exit webcam

    # when everything done, release the capture
    cv2.destroyAllWindows
    return {}
