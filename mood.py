from flask import Flask, request, render_template, jsonify
import cv2
import os
import base64
import numpy as np

app = Flask(__name__)

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def analyze_mood_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mood = "😐 Neutral"
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)

        if len(smiles) > 0:
            mood = "😊 Happy"
        elif len(eyes) >= 2 and h/w < 1:
            mood = "😲 Surprised"
        elif len(eyes) <= 1:
            mood = "😠 Angry"
        else:
            mood = "😐 Neutral"

    return mood

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_image_mood', methods=['POST'])
def detect_image_mood():
    file = request.files.get('image')
    if not file:
        return jsonify({'mood': '❌ No image provided'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    mood = analyze_mood_from_image(img)
    return jsonify({'mood': mood})

@app.route('/detect_webcam_frame', methods=['POST'])
def detect_webcam_frame():
    data_url = request.form['image']
    header, encoded = data_url.split(",", 1)
    img_data = base64.b64decode(encoded)

    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    mood = analyze_mood_from_image(img)
    return jsonify({'mood': mood})

if __name__ == '__main__':
    app.run(debug=True)
