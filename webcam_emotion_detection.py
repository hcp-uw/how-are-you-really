from PIL import Image as im
import numpy as np
from numpy import asarray

# load saved model
from tensorflow.keras.models import model_from_json
model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')

# load Haar-cascade used to detect position of face
import cv2
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# stream webcam video
camera = cv2.VideoCapture(0)
def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

# given image as array of rgba values of each pixel, return predicted emotion
# in detected face and confidence of prediction
def readFace(imgArray):
    # haar_cascade only takes grayscale images; convert into grayscale
    gray_image = cv2.cvtColor(imgArray, cv2.COLOR_BGR2GRAY)
    # detect position of face
    faces = face_haar_cascade.detectMultiScale(gray_image)
    try:
        # TODO: return useful error if no face is detected
        for (x, y, w, h) in faces:
            # tailor image to feed model
            roi_gray = gray_image[y-5:y+h+5, x-5:x+w+5]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = asarray(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0) # convert 3d matrix into 4d tensor
            temp = image_pixels / 255 # normalize
            # use model to predict emotion
            predictions = model.predict(temp)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            return {
                "emotion": emotion_prediction,
                "confidence": str(np.max(predictions[0]))
            }
    except:
        pass

    # TODO: handle error
    return {}