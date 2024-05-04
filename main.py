from flask import Flask, request, render_template, flash, redirect, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import os
import numpy as np
from PIL import Image, ImageOps


model = tf.keras.models.load_model("YOUR_MODEL_NAME")

app = Flask(__name__)
app.config['SECRET_KEY'] = "YOUR_SECRET_KEY"
CORS(app)


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_PATH = "YOUR_FILE_PATH"

def allowed_extensions(filename):
    return '.' in filename and filename.split(".")[-1] in ALLOWED_EXTENSIONS

def get_result(file):
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    img = Image.open(os.path.join(UPLOAD_PATH, file.filename))
    data = np.ndarray(shape=(1,128,128,3), dtype=np.uint8)
    size = (128, 128)
    img = ImageOps.fit(img, size, Image.LANCZOS)
    img_array = np.asarray(img)
    normalized_img_array = (img_array.astype('uint8') / 127.0) - 1
    data[0] = normalized_img_array
    prediction = model.predict(data)
    predicted_class = classes[np.argmax(prediction[0])]
    return predicted_class


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=['POST'])
def predict():
    file = request.files['file']
    if file and allowed_extensions(file.filename):
        file.save(os.path.join(UPLOAD_PATH, file.filename))
        result = get_result(file)
        flash("Image uploaded")
        return render_template('index.html', filename=file.filename, prediction=result)
    else:
        flash("Accepted file types are: png, jpg, jpeg")
        return redirect(request.url)
    
    
@app.route("/display/<filename>")
def display_image(filename):
    return send_from_directory('uploadedFiles', filename)


if __name__ == "__main__":
    app.run(debug=True)