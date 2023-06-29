from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import keras.backend as K
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model('C:/Users/Manan/Desktop/know-your-trash/api/final.h5')

def predict_waste(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    classes = model.predict(x)
    return 'Organic' if classes[0][0] > 0.5 else 'Recyclable'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'})
    if file:
        image_path = 'tmp/image.jpg'
        file.save(image_path)
        result = predict_waste(image_path)
        return jsonify({'result': result})

if __name__ == '__main__':
    app.run()