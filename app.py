from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
from PIL import Image
import base64
import re
import random
import numpy as np

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("cd_resnet50.keras")

@app.route("/", methods=["POST"])
@cross_origin()
def classify():
    img_height, img_width = 384, 512
    class_names = ["cardboard", "glass", "metallic", "paper", "plastic"]
    json = request.get_json()
    imb64str = json["image"]
    imb64str2 = re.sub('^data:image/.+;base64,', '', imb64str)
    imbytes = base64.b64decode(imb64str2)
    with open("image.jpg", "wb") as f:
        f.write(imbytes)
    img = Image.open("image.jpg")
    img = img.resize((img_width, img_height))
    image = np.expand_dims(img, axis=0)
    pred = model.predict(image)
    output = class_names[np.argmax(pred)]
    rand_weight = random.randint(20, 150)
    return jsonify({"trashType": output, "trashWeightGrams": rand_weight})

if __name__ == "__main__":
    app.run(debug=True)
