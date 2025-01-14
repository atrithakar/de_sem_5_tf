from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Load your trained model
model_path = os.path.join(os.getcwd(), 'model1.h5')

model = load_model(model_path)

# Image preprocessing function
def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Assuming the model expects 150x150 input size
    img_array = img_to_array(img)
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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Prepare image for prediction
        image = prepare_image(file_path)

        # Predict the class
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        classes = ['Unripe', 'Ripe', 'Rotten']  # Replace with your model's class names
        
        result = classes[predicted_class]
        
        os.remove(file_path)

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0',port=port,debug=True)
