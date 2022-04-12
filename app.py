from flask import Flask
from flask import render_template, request
import numpy as np
from PIL import Image
import base64
import re
from io import BytesIO


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data_url = request.values['imageBase64']
    # Decoding base64 string to bytes object
    img_bytes = base64.b64decode(re.sub('^data:image/.+;base64,', '', data_url))
    img = Image.open(BytesIO(img_bytes))
    img  = np.array(img)
    return {
        "data": img.tolist()
    }

    """image_b64 = request.values['imageBase64']
    image_b64 = re.sub('^data:image/.+;base64,', '', image_b64).decode('base64')
    image_PIL = Image.open(io.StringIO(image_b64))
    image_np = np.array(image_PIL)
    print('Image received: {}'.format(image_np.shape))
    return image_np
    return {
        "emotion": "happy",
        "confidence": 1.0
    }"""