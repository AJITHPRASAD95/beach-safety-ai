from flask import Flask, request, jsonify
import numpy as np
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

# Load model once
model = YOLO("yolov8n.pt")
model.fuse()

@app.route("/")
def home():
    return "YOLO API running"

@app.route("/esp", methods=["POST"])
def esp():
    try:
        start = time.time()

        # Convert raw bytes → image
        img_array = np.frombuffer(request.data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print("Decode failed")
            return jsonify({"human": 0})

        # Resize for speed (VERY IMPORTANT)
        img = cv2.resize(img, (320, 240))

        # YOLO inference
        results = model(img, verbose=False)

        human = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf > 0.5:
                    human = 1
                    break

        print(f"Result: {human} | Time: {round(time.time()-start,2)}s")

        return jsonify({"human": human})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"human": 0})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
