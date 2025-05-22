import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model

def load_gender_model(model_path):
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def detect_gender(image_path, model_path='gender_detection_model.h5'):
    # Load the model
    model = load_gender_model(model_path)
    if model is None:
        return None, "Error: Could not load model"
    
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error: Could not load image"
    
    # Convert to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect face in the image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if len(faces) == 0:
        return img, "No face detected"
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract face ROI
        face_roi = img_rgb[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = face_roi / 255.0  # Normalize
        face_roi = np.expand_dims(face_roi, axis=0)
        
        # Make prediction
        prediction = model.predict(face_roi)
        gender = "Male" if prediction[0][0] > 0.5 else "Female"
        confidence = prediction[0][0] if gender == "Male" else prediction[0][1]
        
        # Display prediction
        label = f"{gender} ({confidence:.2f})"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    return img, "Face detected"

def detect_gender_realtime():
    # Load the pre-trained model
    model = load_gender_model('gender_detection_model.h5')
    if model is None:
        print("Could not load the model")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract and preprocess face ROI
            face_roi = frame_rgb[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            
            # Make prediction
            prediction = model.predict(face_roi)
            gender = "Male" if prediction[0][0] > 0.5 else "Female"
            confidence = prediction[0][0] if gender == "Male" else prediction[0][1]
            
            # Display prediction
            label = f"{gender} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Gender Detection', frame)
        
        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # For real-time detection
    detect_gender_realtime()
    
    # For single image detection
    # image_path = "test_image.jpg"
    # result_image, message = detect_gender(image_path)
    # if result_image is not None:
    #     cv2.imshow("Gender Detection", result_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     print(message)