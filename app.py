from flask import Flask
from flask import render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analayze():
    return {
        "emotion": "happy",
        "confidence": 1.0
    }