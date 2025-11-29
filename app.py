import io
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image

# -----------------
# API and Model Settings
# -----------------

app = Flask(__name__)

# Load the YOLOv8 model
# This will automatically download yolov8n.pt
MODEL_NAME = 'yolov8n.pt'
model = YOLO(MODEL_NAME)

# Create a temporary folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------
# API Endpoints
# -----------------

@app.route('/', methods=['GET'])
def home():
    """A simple route to check the API's status."""
    return "YOLOv8 Flask API is running! Use /detect endpoint to upload images."

@app.route('/detect', methods=['POST'])
def detect_objects():
    """The route that performs object detection on the uploaded image."""
    
    # Check if an image file has been uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' file provided in the request."}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if file:
        try:
            # 1. Read the image
            # Read the image as bytes using the PIL library
            image_stream = file.read()
            image = Image.open(io.BytesIO(image_stream))
            
            # Convert the PIL image to an OpenCV image (BGR format)
            img_np = np.array(image)
            # RGB -> BGR conversion (required for OpenCV)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 

            # 2. Perform YOLO Inference
            # source=img_cv: Provides the OpenCV format as input
            results = model(img_cv, verbose=False) 

            # 3. Get the results and draw on the image
            # annotated_frame: The image with the detected boxes and labels drawn (BGR format)
            annotated_frame = results[0].plot()

            # 4. Save and return the image
            # Convert the Annotated image back to a PIL image
            img_annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_annotated_rgb)
            
            # Convert the image to bytes and send it in the HTTP response
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)

            return send_file(
                img_byte_arr, 
                mimetype='image/jpeg',
                as_attachment=False,
                download_name='detected_image.jpg'
            )

        except Exception as e:
            # If any error occurs during processing
            print(f"Error during detection: {e}")
            return jsonify({"error": f"An error occurred during detection: {e}"}), 500

if __name__ == '__main__':
    # This can be run for development
    # gunicorn will be used on Render
    app.run(host='0.0.0.0', port=5000)
