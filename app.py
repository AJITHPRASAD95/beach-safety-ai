from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("yolov8n.pt")

@app.route("/predict", methods=["POST"])
def predict():
    img = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    results = model(img, verbose=False)

    human = 0
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # person class
                human = 1
                break

    return jsonify({"human": human})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)