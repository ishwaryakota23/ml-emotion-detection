from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__)
CORS(app)
# Load your trained modelc
model = load_model("final_emotion_model.h5")

# Emotion classes (adjust if you trained with different ones)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    try:
        
        # Forcefully convert image to RGB (3 channels)
        image = Image.open(file).convert('RGB')
        image = image.resize((48, 48))

        # Convert to array
        image = img_to_array(image)  # shape will be (48, 48, 3)
        image = np.expand_dims(image, axis=0)  # shape becomes (1, 48, 48, 3)
        image = image / 255.0  # normalize

        # Predict
        prediction = model.predict(image)
        predicted_class = class_labels[np.argmax(prediction)]

        return jsonify({'emotion': predicted_class})

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
