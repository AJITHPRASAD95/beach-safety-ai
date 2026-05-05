from flask import Flask, request, jsonify
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("yolov8n.pt")

@app.route("/")
def home():
    return "YOLO API running"

@app.route("/esp", methods=["POST"])
def esp():

    try:
        # Read raw bytes
        img_array = np.frombuffer(request.data, np.uint8)

        # Decode image safely
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # 🔴 IMPORTANT CHECK
        if img is None:
            print("❌ Image decode failed")
            return jsonify({"human": 0})

        # Run YOLO
        results = model(img, verbose=False)

        # Detect person class (COCO = 0)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf > 0.5:
                    print("✅ HUMAN DETECTED")
                    return jsonify({"human": 1})

        print("❌ No human")
        return jsonify({"human": 0})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"human": 0})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
