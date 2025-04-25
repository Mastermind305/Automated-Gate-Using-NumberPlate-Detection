
from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import gridfs
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MongoDB connection URI
uri = "mongodb+srv://ag07121321:5G31a9kHwjf5dE6b@cluster0.byjk3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
try:
    client = MongoClient(uri, server_api=ServerApi('1'))
    # Ping the server to check the connection
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    client = None

# Select the database and initialize GridFS
if client:
    db = client['your_database_name']  # Replace with your database name
    fs = gridfs.GridFS(db)
else:
    db = None
    fs = None

# Load your trained YOLO model
try:
    model = YOLO(r"/home/aayush/Desktop/newfolder/platedetection.pt")
    logger.info("Successfully loaded YOLO model")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None

# Initialize the webcam
camera = cv2.VideoCapture(19)
if not camera.isOpened():
    logger.error("Failed to open webcam")

# Set confidence threshold and IoU threshold
conf_threshold = 0.90
iou_threshold = 0.90

frame_count = 0  # To track the number of frames processed

def generate_frames():
    global frame_count
    while True:
        success, frame = camera.read()
        if not success:
            logger.error("Failed to read frame from camera")
            break
        else:
            frame_count += 1
            
            if model:
                # Run YOLO inference on the frame
                results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold)

                # Check if any objects were detected
                if len(results[0].boxes) > 0:
                    # Get current date and time
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    for box in results[0].boxes:
                        # Get the bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Crop the detected region
                        cropped_image = frame[y1:y2, x1:x2]
                        
                        if fs:
                            try:
                                # Save cropped image to MongoDB
                                _, buffer = cv2.imencode('.jpg', cropped_image)  # Encode cropped image as JPEG
                                image_id = fs.put(buffer.tobytes(), 
                                                  filename=f'detected_plate_{frame_count}.jpg',
                                                  metadata={
                                                      'timestamp': current_time,
                                                      'frame_number': frame_count
                                                  })
                                logger.info(f"Detected plate from frame {frame_count} stored with ID: {image_id} at {current_time}")
                            except Exception as e:
                                logger.error(f"Failed to store cropped image in MongoDB: {e}")
                        else:
                            logger.warning("GridFS not initialized, skipping cropped image storage")

                # Draw bounding boxes and labels on the frame
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame
                logger.warning("YOLO model not loaded, displaying raw frame")

            # Add timestamp to the frame
            cv2.putText(annotated_frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Encode the annotated frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        camera.release()
