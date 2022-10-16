import numpy as np
from numpy import asarray

# load saved model
from tensorflow.keras.models import model_from_json
model = model_from_json(open("model/model.json", "r").read())
model.load_weights('model/model.h5')

# load Haar-cascade used to detect position of face
import cv2
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# "database" storing timestamp, emotion, and confidence
database = [
    {"timestamp": "0", "emotion": "happy", "confidence": 10.5},
    {"timestamp": "10", "emotion": "neutral", "confidence": 80.2},
    {"timestamp": "20", "emotion": "happy", "confidence": 12},
    {"timestamp": "30", "emotion": "sad", "confidence": 50},
    {"timestamp": "40", "emotion": "sad", "confidence": 40.5},
    {"timestamp": "50", "emotion": "happy", "confidence": 70},
    {"timestamp": "60", "emotion": "happy", "confidence": 99.9}
]

# predict emotion in detected face in stream webcam video
camera = cv2.VideoCapture(0)
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # haar_cascade only takes grayscale images; convert into grayscale
            gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect position of face
            faces = face_haar_cascade.detectMultiScale(gray_image)
            try:
                for (x, y, w, h) in faces:
                    # draw rectangle around face
                    cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
                    # tailor image to feed model
                    roi_gray = gray_image[y-5:y+h+5, x-5:x+w+5]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    image_pixels = asarray(roi_gray)
                    # convert 3d matrix into 4d tensor
                    image_pixels = np.expand_dims(image_pixels, axis = 0)
                    # normalize
                    temp = image_pixels / 255
                    # use model to predict emotion
                    predictions = model.predict(temp)
                    max_index = np.argmax(predictions[0])
                    emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                    emotion_prediction = emotion_detection[max_index]
                    confidence = str(np.max(predictions[0]))
                    # TODO: save data to "database"
                    # database.append({
                    #     "timestamp": 0, # TODO
                    #     "emotion": emotion_prediction,
                    #     "confidence": confidence
                    # })
                    # TODO: remove?
                    print("The patient is {} with confidence {}".format(emotion_prediction, confidence))
            except:
                pass

            # return frames
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# TODO: enable to configure number of data points and their interval to return
def data():
    return database[-7:]