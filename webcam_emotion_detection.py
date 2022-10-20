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

sessions = {}
session_id = 0
<<<<<<< Updated upstream
analysis_per_session = []
# "database" storing timestamp, emotion, and confidence
database = []

=======
analysis_per_session = {}
>>>>>>> Stashed changes
# predict emotion in detected face in stream webcam video
current_frame = 0

camera = cv2.VideoCapture(0)
def gen_frames():
    global current_frame
    global session_id
    global sessions
    global analysis_per_session
    while True:
        current_frame += 1
        # NOTE: YES! It's called multi[ple times here but while always executes once
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
                    print("The patient is {} with confidence {}".format(emotion_prediction, confidence))

                    # logic for mental state:
                    # 0. for single session: collect emotions in database for every 1 min, 
                    # 1.analyze session, We can keep track of which percent of patients emotions falls under which quadrant. 
                    # Then we can conceptualize these quadrants in different ways: as the dimensions of positive and negative affect, tension and energy, approach and withdrawal, or valence and arousal.
                    # 2. store in another database
                    # 3. show progression over sessions, alert if unusual

                    # ########1. potentially option to start a new session to compare over sessions?
                    # TODO: for now, assuming one session = 60 sec (or 60 analyzed frames)
                    #       later, 1. create a button inside the portal to "Start a new session"
                    #              2. count how many times button was pressed = session_id
                    #              3. each time button is pressed, update session_id in the backend
                    #              4. in UI, have button to stop session and output the analysis to the portal
                    
                    # every minute, start next session
                    # for current session, record emotions under sessions[session_id]
                    # FIXME: this whole file is called once per one frame
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
                        # analyze current session, record under new "database", potentially will fix later
                        # keep track of which percent of patients emotions falls under which quadrant of Russel's Emotional Model
                        # conceptualize these quadrants in different ways: as the dimensions of positive and negative affect, tension and energy, approach and withdrawal, or valence and arousal.
                        for frame_sample in sessions[session_id]:
                            #print("session_id len(sessions[session_id])", session_id, len(sessions[session_id]))
                            if session_id not in analysis_per_session:
                                analysis_per_session[session_id] = {"HV": 0, "LV": 0, "HA": 0, "LA": 0}
                            # 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
                            # The Positive and Negative Affect Schedule (PANAS) 
                            # NA = negative affect, PA = positive affect
                            # high NA is characterized by distress, anger, contempt, and nervousness or fear
                            #            tend to worry excessively about errors and threats, and they may also be sensitive to even minimal stressors
                            # high PA is characterized by joy and high levels of energy, concentration, enthusiasm, and alertness
                            # PA/NA traits reflect the tendency to react negatively or positively to life events
                            # NA is positively correlated with stress related to goal attainment and inter-goal conflicts and 
                            # PA is positively correlated with enjoyment
                            # source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7063053/
                            # Increased levels of negative affect and decreased positive affect are associated with the depressive state or mood
                            
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

                        # NOTE: it won't sump up to 1 as 2 meotions correlate, SO 2 OF THEM SUM UP TO 1
                        # TODO: count # frames in session, then do / (#frames) 
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

                        # skip first session since there's nothing we can compare it against
                        # if len(sessions) > 1:
                        # TODO: compare with prev average
                        session_id +=1
<<<<<<< Updated upstream
                
                    # save data to "database" every second
                    database.append({
                        "timestamp": time.time(),
                        "emotion": emotion_prediction,
                        "confidence": confidence
                    })
                    print("timestamp: {}, emotion: {}, confidence: {}".format(
                        time.time(),
                        emotion_prediction,
                        confidence
                    ))
=======
>>>>>>> Stashed changes
            except:
                pass

            # return frames
<<<<<<< Updated upstream
            ret, buf = cv2.imencode('.jpg', frame)
            frame = buf.tobytes()
=======
            
            ret, buffer = cv2.imencode('.jpg', frame) #FIXME why it fails?
            frame = buffer.tobytes()
>>>>>>> Stashed changes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# return the last entry in "database"
# TODO: configure number of data points and interval to return
def data():
    return database[-1:]