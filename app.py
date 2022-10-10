from flask import Flask, Response
from flask import render_template
import webcam_emotion_detection

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(webcam_emotion_detection.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')