import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import argparse

class RealTimeGenderDetection:
    def __init__(self, model_path='models/gender_detection_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model architecture
        self.model = models.mobilenet_v2(pretrained=False)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 2)
        )
        
        # Load the state dict
        state_dict = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            # If saved as a checkpoint
            self.model.load_state_dict(state_dict['model_state_dict'])
        else:
            # If saved as just the state dict
            self.model.load_state_dict(state_dict)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Define the same transforms used during training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load face detection cascade classifier with better parameters
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Add detection parameters
        self.min_face_size = (60, 60)  # Minimum face size to detect
        self.face_confidence_threshold = 5  # Minimum neighbors for face detection
        self.gender_confidence_threshold = 0.70  # Minimum confidence for gender prediction
        
    def preprocess_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transformations
        tensor_image = self.transform(pil_image)
        return tensor_image.unsqueeze(0).to(self.device)
    
    def detect_gender(self, frame):
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with adjusted parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # Smaller value for more accurate detection
            minNeighbors=self.face_confidence_threshold,  # Higher value to reduce false positives
            minSize=self.min_face_size,  # Minimum face size to detect
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Skip faces that are too large (likely false positives)
            if w > frame.shape[1] * 0.8 or h > frame.shape[0] * 0.8:
                continue
                
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Skip if face ROI is empty
            if face_roi.size == 0:
                continue
                
            # Preprocess face
            try:
                input_tensor = self.preprocess_frame(face_roi)
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    # Only show prediction if confidence is above threshold
                    if confidence >= self.gender_confidence_threshold:
                        # Draw rectangle and label
                        color = (0, 255, 0) if predicted_class == 0 else (255, 0, 0)
                        label = "Male" if predicted_class == 0 else "Female"
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"{label} {confidence*100:.2f}%", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.9, color, 2)
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        return frame
    
    def run_detection(self, source):
        # Handle different types of sources
        if source.startswith('rtsp://'):
            cap = cv2.VideoCapture(source)
        elif source.isdigit():
            cap = cv2.VideoCapture(int(source))  # Webcam
        else:
            cap = cv2.VideoCapture(source)  # Video file
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.detect_gender(frame)
            
            # Display result
            cv2.imshow('Gender Detection', processed_frame)
            
            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Real-time Gender Detection')
    parser.add_argument('--source', type=str, default='0',
                       help='Source (0 for webcam, video file path, or RTSP URL)')
    parser.add_argument('--model', type=str, default='models/gender_detection_model.pt',
                       help='Path to the model file (supports .pt)')
    args = parser.parse_args()
    
    detector = RealTimeGenderDetection(model_path=args.model)
    detector.run_detection(args.source)

if __name__ == "__main__":
    main()