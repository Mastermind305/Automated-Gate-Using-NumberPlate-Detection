
import RPi.GPIO as GPIO
from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import gridfs
from datetime import datetime
import logging
import time

import uuid
from smbprotocol.connection import Connection
from smbprotocol.session import Session
from smbprotocol.tree import TreeConnect
from smbprotocol.open import Open, FilePipePrinterAccessMask, CreateDisposition

#import paramiko
import os
import time
from functools import wraps

#from ftplib import FTP

import threading
import smbclient



server = "192.168.226.117"
#username = "YourWindowsUsername"
#password = "YourWindowsPassword"
share = "pics"

#host = "192.168.226.117"
#port = 22

username = "aayush"
password = "A@yush"
file_name = "captured.jpg"
#remote_dir = "C:/Users/aayush/pics"

#remote_file = f"{remote_dir}/current.jpg"
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GPIO.setmode(GPIO.BOARD)  # Using BCM pin numbering
LED_PIN = 31        # GPIO pin 16 for onboard LED
GPIO.setup(LED_PIN, GPIO.OUT) 
GPIO.output(LED_PIN, GPIO.HIGH)
#for servo
servo_pin = 29 # GPIO pin connected to the servo signal wire
GPIO.setup(servo_pin, GPIO.OUT) 
GPIO.output(servo_pin, GPIO.HIGH)

# Initialize the PWM object




def _close_gate():
    GPIO.output(LED_PIN, GPIO.LOW)
    time.sleep(10)
    GPIO.output(LED_PIN, GPIO.HIGH)
def _open_gate():
    GPIO.output(servo_pin,GPIO.LOW)
    time.sleep(10)
    GPIO.output(servo_pin,GPIO.HIGH)
    print("Gate is CLOSED")

class GateController:
    def __init__(self, open_duration=10):
        self.open_duration = open_duration
        self._remaining_time = 0
        self._timer_lock = threading.Lock()
        self._timer_thread = None
        _close_gate()


    def _manage_timer(self):
        while True:
            with self._timer_lock:
                if self._remaining_time <= 0:
                    _close_gate()
                    self._timer_thread = None
                    break
                self._remaining_time -= 1
            time.sleep(1)

    def trigger_gate(self):
        with self._timer_lock:
            if self._timer_thread is None:
                # Open the gate if it's not already open
                _open_gate()
                self._remaining_time = self.open_duration
                # Start the timer thread
                self._timer_thread = threading.Thread(target=self._manage_timer, daemon=True)
                self._timer_thread.start()
            else:
                # Extend the remaining time
                self._remaining_time = self.open_duration
                print(f"Timer extended. Remaining time: {self._remaining_time} seconds")

gate_controller = GateController(open_duration=40)



def cache_with_timeout(timeout_ms=100):
    """
    Decorator to cache a function's result and skip execution
    if the function is called again within the specified timeout.

    Parameters:
        timeout_ms (int): Timeout in milliseconds within which cached value is returned.

    Returns:
        Decorator function.
    """
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Unique key based on function name, args, and kwargs
            key = (args, frozenset(kwargs.items()))
            current_time = time.time() * 1000  # Convert to milliseconds
            
            # Check if the cache exists and is still valid
            if key in cache:
                last_call_time, cached_result = cache[key]
                if current_time - last_call_time <= timeout_ms:
                    print("Returning cached value")
                    return cached_result
            
            # If not valid, call the function and update the cache
            result = func(*args, **kwargs)
            cache[key] = (current_time, result)
            return result

        return wrapper
    return decorator


    
    


app = Flask(__name__)

# MongoDB connection URI
uri = "mongodb+srv://ag07121321:5G31a9kHwjf5dE6b@cluster0.byjk3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
try:
    #client = uri, server_api=ServerApi('1')
    client = MongoClient(uri, server_api=ServerApi('1'))
    # Ping the server to check the connection
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    client = None
# client = None
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
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    logger.error("Failed to open webcam")

# Set confidence threshold and IoU threshold
conf_threshold = 0.80
iou_threshold = 0.80

frame_count = 0  # To track the number of frames processed

                   
def blink_led():
    
    GPIO.output(LED_PIN, GPIO.LOW)
    time.sleep(5)
    GPIO.output(LED_PIN, GPIO.HIGH)
    GPIO.output(servo_pin,GPIO.LOW)
    time.sleep(5)
    GPIO.output(servo_pin,GPIO.HIGH)
     
#@cache_with_timeout(40) 
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
                @cache_with_timeout(1000)
                def model_loop():
                    # Run YOLO inference on the frame
                    results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold)

                    # Check if any objects were detected
                    if len(results[0].boxes) > 0:
                        gate_controller.trigger_gate()
                        #blink_led()
                        #GPIO.output(LED_PIN, GPIO.LOW)
                        #time.sleep(2)
                        #GPIO.output(LED_PIN, GPIO.HIGH)
                        #GPIO.cleanup()
                        
                        # Get current date and time
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        for box in results[0].boxes:
                        
                            # Get the bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Crop the detected region
                            cropped_image = frame[y1:y2, x1:x2]
                            #success = cv2.imwrite("./current.jpg", cropped_image)
                            # Save cropped image to MongoDB
                            _, imenc = cv2.imencode('.jpg', cropped_image)  # Encode cropped image as JPEG
                            
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            file_name_with_timestamp = f"captured_{timestamp}.jpg"
                            print(f"Generated filename: {file_name_with_timestamp}")
                            
                            img = imenc.tobytes()
                            print("image encoded")
                                            
                            '''try:
                                #Establish SSH client as SFTP connection
                                transport = paramiko.Transport((host, port))
                                transport.connect (username = username, password = password)
                                sftp = paramiko.SFTPClient.from_transport(transport)
                                # Upload the image
                                sftp.put("current.jpg", remote_file_path)
                                print(f"file uploaded to {remote_file_path}")
                                sftp.close()
                                transport.close()
                            except Exception as e:
                                print(f"An error occured: {e}")'''
                            '''
                            try:
                                # Establish SSH Transport
                                transport = paramiko.Transport((host, port))
                                transport.connect(username=username, password=password)

                                # Open SFTP Session
                                sftp = paramiko.SFTPClient.from_transport(transport)

                                # Ensure remote directory exists
                                remote_dir = "C:/Users/aayush/images"
                                try:
                                    sftp.chdir(remote_dir)  # Change to remote directory
                                except IOError:
                                    print(f"Remote directory {remote_dir} does not exist. Creating it.")
                                    sftp.mkdir(remote_dir)  # Create if it doesn’t exist
                                    sftp.chdir(remote_dir)

                                # Upload file
                                sftp.put("current.jpg", remote_dir)
                                print(f"File uploaded to {remote_dir}/current.jpg")

                                # Close SFTP Connection
                                sftp.close()
                                transport.close()

                            except Exception as e:
                                print(f"An error occurred: {e}")
                            
                            '''
                            '''
                            try:
                                ftp = FTP(host)
                                ftp.login(username, password)
                                with open("./current.jpg") as file:
                                    ftp.storbinary("STOR image.jpg", file)
                                    
                                ftp.quit()
                            except Exception as e:
                                print("Error faalyo:", e)
                            '''
                           
                              
                            try:
                                '''conn = Connection(server_name=server, port=445, guid=uuid.uuid4())
                                conn.connect()
                                session = Session(conn, username=username, password=password)
                                session.connect()
                                tree = TreeConnect(session, f"\\\\{server}\\{share}")
                                tree.connect()'''
                                smbclient.ClientConfig(username=username, password=password)
                                remote_file = f"\\\\{server}\\{share}\\{file_name_with_timestamp}"
                            except Exception as e:
                                print("error falyo", e)
                            
                            try:
                                '''file = Open(tree, file_name)
                                #file.create(CreateDisposition.FILE_OVERWRITE_IF, FilePipePrinterAccessMask.GENERIC_WRITE)
                                file.create(
                                        desired_access = FilePipePrinterAccessMask.GENERIC_WRITE,
                                        file_attributes=0,
                                        share_access=0,
                                        create_disposition=CreateDisposition.FILE_OVERWRITE_IF,
                                        create_options=0,
                                        impersonation_level=ImpersonationLevel.Impersonate 
                                        
                                    )

                                #file = Open(tree, file_name, access_mask=FilePipePrinterAccessMask.GENERIC_WRITE)
                                #file.create(CreateDisposition.CREATE_ALWAYS)  # Overwrite if exists
                                #image_data = capture_image()
                                file.write(0, img)
                                file.close()'''
                                with smbclient.open_file(remote_file, mode='wb') as remote_file_handle:
                                    remote_file_handle.write(img)  # Assuming img is the image data as bytes

                                print("File written successfully!")
                            except Exception as e:
                                print("store garda ko error:",e)
                            
                            
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
                    else:
                        print("prediction started but no frame detected")
                    return results[0].plot()

                # Draw bounding boxes and labels on the frame
                annotated_frame = model_loop()
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
    print("requestfromfrontend")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(debug=False)
    finally:
        GPIO.cleanup() 
        camera.release()
