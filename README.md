# SMART PALE DAI 🚗🔐

An **automated gate control system** that uses **YOLOv8 and YOLOv11** for real-time vehicle license plate and character detection, deployed on **Raspberry Pi**, integrated with **MongoDB**, and equipped with a **relay-driven gate actuator**.

## 📌 Project Overview

This system automates the entry of vehicles by:
- Detecting license plates using YOLOv8.
- Performing character recognition using YOLOv11.
- Storing plate data (image, time, and number) in MongoDB.
- Triggering gate opening via GPIO using a relay and linear actuator.
- Displaying detection logs on a user-friendly UI.

## 🎯 Objectives

- 🔍 **License Plate Detection** using YOLOv8.
- 🔠 **Character Recognition** using YOLOv11.
- 🧠 **MongoDB Integration** for record-keeping.
- ⏲ **Smart Timer**: Gate closes automatically after 40 seconds unless another vehicle is detected.

## 🛠 System Architecture

The project is split into two interfaces:
1. **Raspberry Pi Interface (Real-time Detection + Gate Control)**
2. **Desktop Interface (Character Recognition + Data Display)**

## 🧰 Requirements

### Software
- **OS**: Pi OS (Raspberry Pi), Windows 10/11
- **Languages**: Python
- **Libraries**: Ultralytics, TensorFlow/PyTorch, OpenCV, PyMongo, GPIO
- **Tools**: VS Code, Jupyter, Google Colab

### Hardware
- Raspberry Pi 5
- Arduino
- Relay Module
- Webcam
- Linear Actuator

## 📈 Model Performance

| Task                      | Model     | Dataset Size      | mAP Score |
|---------------------------|-----------|--------------------|-----------|
| License Plate Detection   | YOLOv8    | 12,277 images      | 98.8%     |
| Character Detection       | YOLOv11   | 936 plate images   | 94.5%     |

## 🧪 Methodology

### License Plate Detection
- Live webcam feed processed using YOLOv8
- Bounding boxes extracted and saved
- Plate images stored in MongoDB

### Gate Opening Logic
- If a plate is detected, GPIO pin is HIGH (Gate opens)
- A 40-second timer starts
- If a new vehicle is detected within this window, the timer resets
- Gate closes if no vehicle is detected after 40 seconds

### Character Detection & UI
- UI fetches frames from MongoDB
- YOLOv11 detects characters
- Results displayed in tabular format: serial number, plate number, image, timestamp

## 🚀 Deployment

- YOLO models run independently on Raspberry Pi and Desktop
- Gate hardware integrated via relay circuit
- MongoDB ensures consistent and organized data flow between systems

## 📝 License

This project is for academic purposes. Feel free to fork and build upon it for research or development.
