import cv2
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import time

class GenderDetector:
    def __init__(self, model_path='models/gender_detection_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)
        
        # Load model weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
        
        # Setup face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Setup image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_face(self, face_roi):
        # Convert BGR to RGB and then to PIL Image
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # Apply transformations
        face_tensor = self.transform(face_pil)
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        return face_tensor
    
    def predict_gender(self, face_tensor):
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            gender = "Male" if predicted.item() == 1 else "Female"
            confidence = probabilities[0][predicted].item()
            
            return gender, confidence
    
    def process_frame(self, frame):
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Draw rectangle around each face and predict gender
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face_roi = frame[y:y+h, x:x+w]
            face_tensor = self.preprocess_face(face_roi)
            
            # Predict gender
            gender, confidence = self.predict_gender(face_tensor)
            
            # Draw rectangle and label
            color = (0, 255, 0) if gender == "Female" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Add prediction text
            label = f"{gender}: {confidence:.2%}"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame, len(faces)

    def detect_from_image(self, image_path):
        """Detect gender from a single image file"""
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Could not load image")
        
        # Process frame
        result_frame, num_faces = self.process_frame(frame)
        
        return result_frame, num_faces
    
    def detect_from_video(self, video_path, output_path=None):
        """Detect gender from a video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video source")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Setup video writer if output path is specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (frame_width, frame_height))
        
        frame_count = 0
        processing_times = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame and measure time
                start_time = time.time()
                processed_frame, num_faces = self.process_frame(frame)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Add FPS info
                fps_text = f"FPS: {1/processing_time:.1f}"
                cv2.putText(processed_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame if output path is specified
                if output_path:
                    out.write(processed_frame)
                
                # Display frame
                cv2.imshow('Gender Detection', processed_frame)
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
        
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
        
        # Return processing statistics
        avg_fps = 1 / (sum(processing_times) / len(processing_times))
        return {
            'frames_processed': frame_count,
            'average_fps': avg_fps,
            'total_time': sum(processing_times)
        }
    
    def detect_from_webcam(self):
        """Real-time gender detection from webcam"""
        return self.detect_from_video(0)  # 0 is webcam index

def main():
    # Create detector instance
    detector = GenderDetector()
    
    # Example usage menu
    while True:
        print("\nGender Detection Menu:")
        print("1. Real-time Webcam Detection")
        print("2. Image File Detection")
        print("3. Video File Detection")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            print("Starting webcam detection... Press 'q' to quit")
            stats = detector.detect_from_webcam()
            print(f"\nProcessing Statistics:")
            print(f"Frames Processed: {stats['frames_processed']}")
            print(f"Average FPS: {stats['average_fps']:.1f}")
            
        elif choice == '2':
            image_path = input("Enter the path to image file: ")
            try:
                result_frame, num_faces = detector.detect_from_image(image_path)
                print(f"Detected {num_faces} faces")
                
                cv2.imshow('Gender Detection', result_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"Error processing image: {str(e)}")
            
        elif choice == '3':
            video_path = input("Enter the path to video file: ")
            save_output = input("Save output video? (y/n): ").lower() == 'y'
            
            output_path = None
            if save_output:
                output_path = input("Enter output video path (e.g., output.mp4): ")
            
            try:
                stats = detector.detect_from_video(video_path, output_path)
                print(f"\nProcessing Statistics:")
                print(f"Frames Processed: {stats['frames_processed']}")
                print(f"Average FPS: {stats['average_fps']:.1f}")
                print(f"Total Processing Time: {stats['total_time']:.1f} seconds")
                
            except Exception as e:
                print(f"Error processing video: {str(e)}")
            
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()