from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = load_model('lung_cancer_model_vgg.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        file_path = f'static/{file.filename}'
        file.save(file_path)

        img_array = preprocess_image(file_path)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        
        class_labels = ['Benign', 'Malignant', 'Normal']
        predicted_label = class_labels[predicted_class]

        return render_template('index.html', prediction_result=predicted_label, image_path=file_path)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
