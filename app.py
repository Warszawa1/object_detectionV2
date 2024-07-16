import ssl
from flask import Flask, render_template, request, Response
import cv2
import numpy as np
from ultralytics import YOLO

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

app = Flask(__name__)

def process_frame(frame):
    # Calculate aspect ratio
    aspect_ratio = frame.shape[1] / frame.shape[0]
    
    # Set a maximum width for processing (adjust as needed)
    max_width = 800
    
    # Resize image while maintaining aspect ratio
    if frame.shape[1] > max_width:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height))
    else:
        resized_frame = frame
    
    # Process with YOLO
    results = model(resized_frame)
    
    # Draw results on the original frame to maintain quality
    annotated_frame = results[0].plot()
    
    # Resize back to original size if it was resized
    if frame.shape[1] > max_width:
        annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))
    
    return annotated_frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
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
    return "Invalid file"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)