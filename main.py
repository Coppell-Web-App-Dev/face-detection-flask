from flask import Flask, render_template, request
import numpy as np
import cv2, base64


app = Flask(__name__)

classifier_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(classifier_path)

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/post_image', methods=['POST', 'GET'])
def post_image():
    if request.method == 'POST':
        image_file = request.files.get('image-file', '')
        filestr = image_file.read()
        npimg = np.fromstring(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = detect(image)
        retval, buffer = cv2.imencode('.png', image)
        data_uri = base64.b64encode(buffer).decode('ascii')
        return render_template('display.html', image=image)

if __name__ == "__main__":
    app.run(debug=True)
