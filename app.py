from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__)

model_detection = YOLO("MyOutfitApp_trained_best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['file']
        image = Image.open(image_file)


        results = model_detection(image)


        detected_items = [model_detection.names[int(box.cls)] for box in results[0].boxes]


        label_counts = Counter(detected_items)

        formatted_detected_items = ', '.join([f"{count} {label}" for label, count in label_counts.items()])

        return jsonify({"detected_items": formatted_detected_items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500