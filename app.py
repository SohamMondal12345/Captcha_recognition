from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from tensorflow import keras

# Define constants
IMG_SHAPE = (50, 200, 1)
SYMBOLS = list(map(chr, range(97, 123))) + list(map(chr, range(48, 58)))  # a-z + 0-9
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = keras.models.load_model('captcha_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = image / 255.0
    image = image.reshape(IMG_SHAPE)
    return np.expand_dims(image, axis=0)

def predict_captcha(image):
    results = model.predict(image)
    predicted_labels = [SYMBOLS[np.argmax(results[i])] for i in range(len(results))]
    return ''.join(predicted_labels)

@app.route('/')
def index():
    return render_template('index.html')
   

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image = preprocess_image(file_path)
        captcha_text = predict_captcha(image)
        return jsonify({'captcha_text': captcha_text})
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
