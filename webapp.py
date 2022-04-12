from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)
LoadedModel = load_model(os.path.join("flowerClassification.model"))


def prediction(path):
    labels = {'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}
    testImage = cv2.imread(path)
    image = cv2.resize(testImage, (224, 224))
    image = image / 255
    classes = LoadedModel.predict(image.reshape(1, 224, 224, 3))
    for label, index in labels.items():
        if index == np.argmax(classes[1]):
            return label, round(np.max(classes[1]),3) * 100
        else:
            return "NOT FOUND", 0


@app.route("/", methods=["POST", "GET"])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == "POST":
        image = request.files.get('imgup')
        image.save('./' + secure_filename(image.filename))
        name, score = prediction(image.filename)
        kwargs = {'name': name, 'score': score}
        return render_template('index2.html', **kwargs)


if __name__ == '__main__':
    app.run()
