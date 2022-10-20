import numpy as np
from numpy import asarray
import time

# load saved model
from tensorflow.keras.models import model_from_json
model = model_from_json(open("model/model.json", "r").read())
model.load_weights('model/model.h5')

# load Haar-cascade used to detect position of face
import cv2
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# mental state analysis "databases"
sessions = {}
session_id = 0
analysis_per_session = {}
current_frame = 0

# "database" storing timestamp, emotion, and confidence
database = []

camera = cv2.VideoCapture(0)
def gen_frames():
    global current_frame
    global session_id
    global sessions
    global analysis_per_session
    while True:
        current_frame += 1
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
                    # save data to "database" every second
                    database.append({
                        "timestamp": time.time(),
                        "emotion": emotion_prediction,
                        "confidence": confidence
                    })
                    print("The patient is {} with confidence {}".format(emotion_prediction, confidence))

                    if current_frame % 10 != 0:

                        if session_id not in sessions:
                            sessions[session_id] = []

                        sessions[session_id].append ({
                            "timestamp": time.time(),
                            "emotion": emotion_prediction,
                            "confidence": confidence
                        })
                    else:
                        sessions[session_id].append ({
                            "timestamp": time.time(),
                            "emotion": emotion_prediction,
                            "confidence": confidence
                        })
                        for frame_sample in sessions[session_id]:
                            if session_id not in analysis_per_session:
                                analysis_per_session[session_id] = {"HV": 0, "LV": 0, "HA": 0, "LA": 0}

                            # high arousall (i.e. intensity) + LV (low valence i.e. balance of positivity/neagtivity) = anxiety
                            # low arousal + LV (low valence) = depression
                            # high arousal + HV (high valence) = alertness, enthusiasm
                            # low arousal + HV (high valence) = enjoyment, calm positive state
                            # 'happy' = HA + HV, 'sad' = LA + LV, 'neutral' = LA + HV, 'angry' = HA + LV, 'disgust' = LV + HA 
                            if frame_sample["emotion"] == 'sad':
                                analysis_per_session[session_id]["LA"] += 1
                                analysis_per_session[session_id]["LV"] += 1
                            elif frame_sample["emotion"] == 'happy':
                                analysis_per_session[session_id]["HA"] += 1
                                analysis_per_session[session_id]["HV"] += 1
                            elif frame_sample["emotion"] == 'neutral':
                                analysis_per_session[session_id]["LA"] += 1
                                analysis_per_session[session_id]["HV"] += 1
                            elif frame_sample["emotion"] == 'fear':
                                analysis_per_session[session_id]["LA"] += 1
                                analysis_per_session[session_id]["LV"] += 1
                            elif frame_sample["emotion"] == 'angry':
                                analysis_per_session[session_id]["HA"] += 1
                                analysis_per_session[session_id]["LV"] += 1
                            elif frame_sample["emotion"] == 'disgust':
                                analysis_per_session[session_id]["HA"] += 1
                                analysis_per_session[session_id]["LV"] += 1    

                        analysis_per_session[session_id]["LA"] = round(analysis_per_session[session_id]["LA"] / 10, 3)
                        analysis_per_session[session_id]["HA"] = round(analysis_per_session[session_id]["HA"] / 10, 3)

                        analysis_per_session[session_id]["LV"] = round(analysis_per_session[session_id]["LV"] / 10, 3)
                        analysis_per_session[session_id]["HV"] = round(analysis_per_session[session_id]["HV"] / 10, 3)

                        print("#########################################################")
                        print("{} of patient's emotional state falls under LA".format(analysis_per_session[session_id]["LA"]))
                        print("{} of patient's emotional state falls under HA".format(analysis_per_session[session_id]["HA"]))
                        print("{} of patient's emotional state falls under LV".format(analysis_per_session[session_id]["LV"]))
                        print("{} of patient's emotional state falls under HV".format(analysis_per_session[session_id]["HV"]))
                        print("#########################################################")

                        session_id +=1
            except:
                pass

            # return frames
            
            ret, buffer = cv2.imencode('.jpg', frame) 
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# return the last entry in "database"
# TODO: configure number of data points and interval to return
def data():
    return database[-1:]