from flask import Flask, Response
from flask import render_template, jsonify
import webcam_emotion_detection

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
<<<<<<< HEAD
    # NOTE: not here, called only once,in the very very beginning
    return Response(webcam_emotion_detection.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
=======
    return Response(webcam_emotion_detection.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/data")
def data():
    return jsonify(webcam_emotion_detection.data())
>>>>>>> origin/main
