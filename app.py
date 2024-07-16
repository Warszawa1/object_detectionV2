import ssl
import certifi
from flask import Flask, render_template, request, Response
import cv2
import numpy as np
from ultralytics import YOLO

# Set default SSL context to use certifi certificates
ssl._create_default_https_context = ssl._create_unverified_context

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

app = Flask(__name__)

def process_frame(frame):
    # Perform object detection
    results = model(frame)

    # Render the results on the frame
    annotated_frame = results[0].plot()

    # Check for specific objects (e.g., 'dog')
    for result in results:
        for label in result.names:
            if label == "dog":
                print("Dog detected!")  # This is where you can trigger a notification

    return annotated_frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    print("Received request to /video_feed")
    print("Request method:", request.method)
    print("Request form data:", request.form)
    print("Request files:", request.files)
    if 'file' not in request.files:
        print("No file part")
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return "No selected file"
    if file:
        print("Processing file")
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    print("Invalid file")
    return "Invalid file"

@app.route('/upload_test', methods=['GET', 'POST'])
def upload_test():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            processed_frame = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    return render_template ("test_upload.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)