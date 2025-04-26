from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Ensure 'uploads' directory exists
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


model = load_model("plant_disease_model.h5")


import os

dataset_path = r"C:\Users\samar\Desktop\Contriver\Plant_disease_prediction\dataset"
class_labels = sorted(os.listdir(dataset_path))

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128,3))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return class_labels[class_index]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        predicted_disease = predict_disease(file_path)

        return render_template("result.html", image_url=file_path, prediction=predicted_disease)

    return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)
