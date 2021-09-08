from flask import Flask, request, Response, render_template, current_app
import sys
import os
import wave
import subprocess
import json
from flask_sockets import Sockets
import base64
#import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from matplotlib.patches import Rectangle
from test_api import model_eval, request_speech

MAX_TARGET_LEN   = 100
PAD_LEN          = 1600
TEST_BATCH_SIZE  = 3
NUM_HID          = 512
NUM_HEAD         = 4
FEED_FORWARD     = 1024
MODEL_           = "./model/checkpoint_17"


def basename(filename):
    return filename.rsplit('.', 1)[0]


# FORMAT = pyaudio.paInt16
# CHANNELS = 2
# RATE = 44100
# CHUNK = 1024
# RECORD_SECONDS = 5

#audio1 = pyaudio.PyAudio()
model, vectorizer = model_eval(MODEL_, MAX_TARGET_LEN, FEED_FORWARD, NUM_HEAD, NUM_HID)



app = Flask(__name__,static_url_path='/static')

@app.route("/stt", methods=["GET","POST"])
def stt():
    if request.method == "POST":
        print(request)
        print("INPUT")
        file=request.files["file"]
        file_base = basename(file.filename)
        file.save("test.wav")
        subprocess.check_call(args=['ffmpeg', '-i','{}'.format("test.wav"),\
                 '-acodec','pcm_s16le','-ac','1', '-ar', '16000','{}'.format("e_" + "test.wav"), '-y'])

        output = request_speech(model, "e_test.wav", PAD_LEN, vectorizer)
        os.remove("test.wav")
        os.remove("e_" + "test.wav")
        
        return output

    elif request.method == "GET":
        return render_template("index.html")

@app.route("/")
def index():
    return "Anasayfa."
