from flask import Flask
from flask import render_template, request
import numpy as np
from PIL import Image
import base64
import re
from io import BytesIO
import webcam_emotion_detection

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data_url = request.values["imageBase64"]
    # decode base64 string to bytes object
    img_bytes = base64.b64decode(re.sub("^data:image/.+;base64,", "", data_url))
    img = Image.open(BytesIO(img_bytes))
    img = np.array(img)
    data = webcam_emotion_detection.readFace(img)
    return data