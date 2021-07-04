from flask import Flask, request, render_template
import os

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
from imutils import paths
import base64
import uuid

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LABEL_BIN = os.path.join(BASE_DIR, "label_binary.pickle")
MODEL_PATH = os.path.join(BASE_DIR, "mashroom_recog_new.model")
IMG_TEMP_STORAGE = os.path.join(BASE_DIR, "teporary_image_storage") # you can delete manually

model = load_model(MODEL_PATH)
lb = pickle.loads(open(LABEL_BIN, "rb").read())

# Mashrooms = [
#     'Amanita_Caesarea-Edible',
#     'Amanita_Citrina-Edible',
#     'Amanita_Pantherina-NotEdible',
#     'Boletus_Regius-Edible',
#     'Clitocybe_Costata-Edible',
#     'Entoloma_Lividum-NotEdible',
#     'Gyromitra_Esculenta-NotEdible',
#     'Helvella_Crispa-Edible',
#     'Hydnum_Rufescens-NotEdible',
#     'Hygrophorus_Latitabundus-Edible',
#     'Morchella_Deliciosa-Edible',
#     'Omphalotus_Olearius-NotEdible',
#     'Phallus_Impudicus-NotEdible',
#     'Rubroboletus_Satanas-NotEdible',
#     'Russula_Cyanoxantha-Edible',
#     'Russula_Delica-NotEdible'
# ]

if not os.path.exists(IMG_TEMP_STORAGE):
    os.mkdir(IMG_TEMP_STORAGE)


@app.get('/')
def get_image():
    return render_template('index.html')

@app.post('/predict_result')
def predict_images():
    data = request.get_json()
    encoded_data = data['image_data'].split(',')[1]
    nparr = np.frombuffer(
        base64.b64decode(encoded_data), np.uint8
    )
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print(img)
    temp_storage_name = f"{ str(uuid.uuid4()) }.png"
    path_to_temp_storage = os.path.join(IMG_TEMP_STORAGE, temp_storage_name)
    print(path_to_temp_storage)
    cv2.imwrite(path_to_temp_storage, img) # saving to temp_storage
    image = cv2.imread(path_to_temp_storage)
    image = cv2.resize(image, (224, 224))
    # image = cv2.dnn.blobFromImage(image, 1, (224, 224), (103.939, 116.779, 123.68))
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    image = image - mean
    image = np.expand_dims(image, axis=0)
    image = image.reshape(1,224,224,3)
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    print(lb.classes_)
    new_label = lb.classes_[idx]
    edibility = new_label.split('-')[1]
    if edibility == "Edible":
        edibility = "Edible"
    else:
        edibility = "Not Edible"
    # new_label = "Predicted Label: {} with {:.2f} confidence%".format(new_label, proba[idx] * 100)
    class_label = new_label.split('-')[0].split('_')
    class_label = f"{class_label[0]}  {class_label[1]}"
    print(new_label)
    return {
        "status_code": 200,
        "class": class_label,
        "confidence": proba[idx] * 100,
        "edibility": edibility
    }


if __name__ == "__main__":
    app.run(debug=True)
