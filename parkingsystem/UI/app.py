# from flask import Flask, render_template, jsonify, send_file
# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi
# import gridfs
# from bson import ObjectId
# import io
# import logging
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import base64
# from datetime import datetime
# from pymongo.errors import ConnectionFailure

# app = Flask(__name__)

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)

# # MongoDB connection
# uri = ""
# client = MongoClient(uri, server_api=ServerApi('1'))
# db = client['your_database_name']
# fs = gridfs.GridFS(db)

# # Test database connection
# try:
#     client.admin.command('ping')
#     app.logger.info("Successfully connected to MongoDB")
# except ConnectionFailure:
#     app.logger.error("Failed to connect to MongoDB")

# # Load YOLO model
# def load_character_detection_model():
#     character_model_path = r"C:\Users\Asus\Downloads\characterdetectionv11.pt"  # Update path as needed
#     return YOLO(character_model_path)

# character_model = load_character_detection_model()
# character_classes = [
#     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'BA', 'BAGMATI', 'CHA',
#     'GA', 'GANDAKI', 'HA', 'JA', 'JHA', 'KA', 'KHA', 'KO', 'LU', 'LUMBINI',
#     'MA', 'MADESH', 'ME', 'NA', 'PA', 'PRA', 'PRADESH', 'RA', 'SU', 'VE', 'YA'
# ]

# # ... (keep the existing functions: calculate_dynamic_threshold, optimized_sort_characters, process_image)

# def store_vehicle_data(numberplate, image_data, entry_time):
#     try:
#         # Check if the numberplate already exists
#         existing_vehicle = db.vehicles.find_one({"numberplate": numberplate})
#         if existing_vehicle:
#             # If it exists, update the entry time and image
#             image_id = fs.put(image_data, filename=f"{numberplate}.jpg")
#             db.vehicles.update_one(
#                 {"numberplate": numberplate},
#                 {"$set": {
#                     "image_id": image_id,
#                     "entry_time": entry_time
#                 }}
#             )
#         else:
#             # If it's a new entry, insert it
#             image_id = fs.put(image_data, filename=f"{numberplate}.jpg")
#             db.vehicles.insert_one({
#                 "numberplate": numberplate,
#                 "image_id": image_id,
#                 "entry_time": entry_time
#             })
#         return True, image_id
#     except Exception as e:
#         app.logger.error(f"Error storing vehicle data: {str(e)}")
#         return False, None

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/get_images")
# def get_images():
#     try:
#         # Fetch unique vehicles from the vehicles collection
#         vehicles = list(db.vehicles.find().sort("entry_time", -1))
        
#         image_list = []
#         for index, vehicle in enumerate(vehicles, start=1):
#             try:
#                 image_data = fs.get(vehicle['image_id']).read()
#                 detected_text, processed_image = process_image(image_data)
                
#                 image_list.append({
#                     'vehicle_no': index,
#                     'numberplate': detected_text,
#                     'entry_time': vehicle['entry_time'].isoformat(),
#                     'processed_image': processed_image
#                 })
#             except gridfs.errors.NoFile:
#                 app.logger.error(f"No file found in GridFS for vehicle {vehicle['numberplate']}")
#                 # Skip this vehicle and continue with the next one
#                 continue
        
#         app.logger.info(f"Found and processed {len(image_list)} unique vehicles")
#         return jsonify(image_list)
#     except Exception as e:
#         app.logger.error(f"Error in get_images: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route("/download_image/<image_id>")
# def download_image(image_id):
#     try:
#         file_id = ObjectId(image_id)
#         image_file = fs.get(file_id)
#         file_data = io.BytesIO(image_file.read())
#         file_data.seek(0)
        
#         return send_file(
#             file_data,
#             mimetype='image/jpeg',
#             as_attachment=True,
#             download_name=image_file.filename
#         )
#     except gridfs.errors.NoFile:
#         app.logger.error(f"No file found in GridFS with id {image_id}")
#         return jsonify({'error': 'Image not found'}), 404
#     except Exception as e:
#         app.logger.error(f"Error in download_image: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)




from flask import Flask, render_template, jsonify, send_file
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import gridfs
from bson import ObjectId
import io
import logging
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from datetime import datetime

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# MongoDB connection
uri = "mongodb+srv://ag07121321:5G31a9kHwjf5dE6b@cluster0.byjk3.mongodb.net/"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['your_database_name']
fs = gridfs.GridFS(db)

# Load YOLO model
def load_character_detection_model():
    character_model_path = r"C:\Users\Asus\Downloads\characterdetectionv11.pt"  # Update path as needed
    return YOLO(character_model_path)

character_model = load_character_detection_model()
character_classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'BA', 'BAGMATI', 'CHA',
    'GA', 'GANDAKI', 'HA', 'JA', 'JHA', 'KA', 'KHA', 'KO', 'LU', 'LUMBINI',
    'MA', 'MADESH', 'ME', 'NA', 'PA', 'PRA', 'PRADESH', 'RA', 'SU', 'VE', 'YA'
]



def calculate_dynamic_threshold(characters, num_lines, min_threshold=15, max_threshold=50):
    if len(characters) < 2:
        return max_threshold
    horizontal_distances = [abs(characters[i]['x'] - characters[i - 1]['x']) for i in range(1, len(characters))]
    avg_distance = np.mean(horizontal_distances)
    line_adjustment = np.clip(100 / (num_lines + 1), 15, 30)
    return np.clip(avg_distance / 2 + line_adjustment, min_threshold, max_threshold)

def optimized_sort_characters(characters, dynamic_threshold=True):
    characters.sort(key=lambda x: x['y'])
    lines, current_line = [], []
    threshold = calculate_dynamic_threshold(characters, len(set(round(char['y']) for char in characters))) if dynamic_threshold else 25
    for char in characters:
        if not current_line or abs(current_line[-1]['y'] - char['y']) < threshold:
            current_line.append(char)
        else:
            lines.append(sorted(current_line, key=lambda x: x['x']))
            current_line = [char]
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x['x']))
    return sorted(lines, key=lambda line: line[0]['y'])

def process_image(image_data):
    nparr = np.frombuffer(image_data, np.uint8)
    extracted_plate = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray_image = cv2.cvtColor(extracted_plate, cv2.COLOR_BGR2GRAY)
    gray_image_resized = cv2.resize(gray_image, (640, 640))
    gray_image_3channel = cv2.cvtColor(gray_image_resized, cv2.COLOR_GRAY2BGR)

    results = character_model(gray_image_3channel, conf=0.5)
    detected_classes = results[0].boxes.cls.cpu().numpy()
    boxes = results[0].boxes.xywh.cpu().numpy()

    characters = [
        {'class': character_classes[int(cls)], 'x': box[0], 'y': box[1], 'w': box[2], 'h': box[3]}
        for cls, box in zip(detected_classes, boxes)
        if int(cls) < len(character_classes)
    ]

    sorted_characters = optimized_sort_characters(characters)
    sorted_string = ''.join([char['class'] for line in sorted_characters for char in line])

    for char in characters:
        x, y, w, h = int(char['x']), int(char['y']), int(char['w']), int(char['h'])
        cv2.rectangle(gray_image_resized, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 0, 0), 2)
        cv2.putText(gray_image_resized, char['class'], (x - w // 2, y - h // 2 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    _, buffer = cv2.imencode('.jpg', gray_image_resized)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return sorted_string, img_base64

# ... (keep all the existing functions: calculate_dynamic_threshold, optimized_sort_characters, process_image)







def migrate_data_to_vehicles():
    try:
        # Get all files from fs.files that are not in vehicles collection
        all_files = db.fs.files.find()
        for file in all_files:
            # Check if this file is already in vehicles collection
            existing_vehicle = db.vehicles.find_one({"image_id": file['_id']})
            if not existing_vehicle:
                # Process the image to get the numberplate
                image_data = fs.get(file['_id']).read()
                numberplate, _ = process_image(image_data)
                
                # Store in vehicles collection
                db.vehicles.insert_one({
                    "numberplate": numberplate,
                    "image_id": file['_id'],
                    "entry_time": file.get('uploadDate', datetime.utcnow())
                })
        
        app.logger.info("Data migration completed successfully")
    except Exception as e:
        app.logger.error(f"Error in data migration: {str(e)}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_images")
def get_images():
    try:
        # Migrate data from fs.files to vehicles collection
        migrate_data_to_vehicles()
        
        # Fetch unique vehicles from the vehicles collection
        vehicles = db.vehicles.find().sort("entry_time", -1)
        
        image_list = []
        for index, vehicle in enumerate(vehicles, start=1):
            image_data = fs.get(vehicle['image_id']).read()
            detected_text, processed_image = process_image(image_data)
            
            image_list.append({
                'vehicle_no': index,
                'numberplate': detected_text,
                'entry_time': vehicle['entry_time'].isoformat(),
                'processed_image': processed_image
            })
        
        app.logger.info(f"Found and processed {len(image_list)} unique vehicles")
        return jsonify(image_list)
    except Exception as e:
        app.logger.error(f"Error in get_images: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route("/download_image/<image_id>")
def download_image(image_id):
    try:
        file_id = ObjectId(image_id)
        image_file = fs.get(file_id)
        file_data = io.BytesIO(image_file.read())
        file_data.seek(0)
        
        return send_file(
            file_data,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=image_file.filename
        )
    except gridfs.errors.NoFile:
        app.logger.error(f"No file found in GridFS with id {image_id}")
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        app.logger.error(f"Error in download_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)    




 

