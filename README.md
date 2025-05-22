# Gender Detection System using Deep Learning

This project implements a gender detection system using computer vision and deep learning (TensorFlow/Keras). It can detect gender from images, video files, webcam, or RTSP streams, and save detection results to a text file.

## Features
- Train a gender classification model using the UTKFace dataset
- Detect gender in single images
- Real-time gender detection using webcam
- Gender detection from video files or RTSP streams
- Save detection results (frame, bounding box, gender, confidence) to a text file

## Project Structure
```
gender_detection_model.h5         # Trained model (generated after training)
gender_detection.py               # Main script for detection
train_model.py                    # Script to train the model
requirements.txt                  # Python dependencies
UTKFace/                          # Dataset (with part1, part2, part3 subfolders)
README.md                         # This file
```

## Setup Instructions

1. **Clone the repository and navigate to the project directory**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download and extract the UTKFace dataset**
- Place the dataset in the `UTKFace/` directory, with images inside `part1/`, `part2/`, and `part3/` subfolders.

## Training the Model

To train the gender detection model on the UTKFace dataset:
```bash
python train_model.py
```
- The trained model will be saved as `gender_detection_model.h5` in the project directory.

## Using the Gender Detection System

### 1. Detect Gender in a Single Image
- Edit `gender_detection.py` and uncomment the single image detection section in the `__main__` block:
```python
image_path = "test_image.jpg"
result_image, message = detect_gender(image_path)
if result_image is not None:
    cv2.imshow("Gender Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(message)
```
- Run:
```bash
python gender_detection.py
```

### 2. Real-Time Gender Detection (Webcam)
- Uncomment the following in `gender_detection.py`:
```python
detect_gender_realtime()
```
- Run:
```bash
python gender_detection.py
```
- Press 'q' to quit the webcam window.

### 3. Gender Detection on Video File or RTSP Stream
- Uncomment and edit the following in `gender_detection.py`:
```python
# For video file:
detect_gender_video("test_video.mp4", "detection_results.txt")
# For RTSP stream:
detect_gender_video("rtsp://username:password@ip_address:port/stream", "detection_results.txt")
```
- Run:
```bash
python gender_detection.py
```
- Detection results will be saved in `detection_results.txt` with columns: frame, x, y, w, h, gender, confidence.

## Notes
- The model expects face images of size 64x64x3 (RGB).
- The UTKFace dataset filenames encode gender as 0 (male) and 1 (female).
- The system uses OpenCV's Haar Cascade for face detection.
- For best results, ensure good lighting and clear face visibility in input images/videos.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- numpy
- scikit-learn
- keras

Install all requirements with:
```bash
pip install -r requirements.txt
```

## License
This project is for educational and research purposes.