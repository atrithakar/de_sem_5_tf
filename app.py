from flask import Flask, render_template, request, redirect, url_for
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model_path = os.path.join(os.getcwd(), 'model1.h5')
model = load_model(model_path)

# Image preprocessing function
def prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize to match the model's expected input size
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        try:
            # Read the image in memory
            image = Image.open(BytesIO(file.read()))
            
            # Prepare the image for prediction
            image = prepare_image(image)

            # Predict the class
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            classes = ['Unripe', 'Ripe', 'Rotten']  # Replace with your model's class names
            
            result = classes[predicted_class]
        except Exception as e:
            # Handle exceptions (e.g., invalid image format)
            print(f"Error processing image: {str(e)}")

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
