import cv2
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import time
import mediapipe as mp
import urllib.parse

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
        
        # Setup MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        # Setup image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def extract_face_roi(self, frame, detection):
        """Extract face ROI using MediaPipe detection"""
        ih, iw, _ = frame.shape
        bbox = detection.location_data.relative_bounding_box
        
        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)
        
        # Add padding to include more context
        padding_x = int(w * 0.1)
        padding_y = int(h * 0.1)
        
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(iw, x + w + padding_x)
        y2 = min(ih, y + h + padding_y)
        
        return frame[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)
    
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
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        num_faces = 0
        if results.detections:
            for detection in results.detections:
                # Extract face ROI
                face_roi, (x, y, w, h) = self.extract_face_roi(frame, detection)
                
                if face_roi.size == 0:  # Skip if face ROI is empty
                    continue
                
                # Predict gender
                face_tensor = self.preprocess_face(face_roi)
                gender, confidence = self.predict_gender(face_tensor)
                
                # Draw rectangle and label
                color = (0, 255, 0) if gender == "Female" else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Add prediction text with nicer formatting
                label = f"{gender}: {confidence:.1%}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y-25), (x + label_size[0], y), color, -1)
                cv2.putText(frame, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                num_faces += 1
        
        return frame, num_faces

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

    def detect_from_stream(self, stream_url, output_path=None):
        """Detect gender from a video stream (RTSP or HTTP)"""
        # Validate URL
        try:
            parsed = urllib.parse.urlparse(stream_url)
            if parsed.scheme not in ['rtsp', 'http', 'https']:
                raise ValueError("Invalid stream URL. URL must start with 'rtsp://', 'http://', or 'https://'")
        except Exception as e:
            raise ValueError(f"Invalid stream URL: {str(e)}")
        
        print(f"Connecting to stream: {stream_url}")
        
        # Open stream
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise ValueError("Could not connect to stream")
        
        # Configure capture buffer size
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        # Get stream properties
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
        retry_count = 0
        max_retries = 5
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    retry_count += 1
                    print(f"Stream error, retry {retry_count}/{max_retries}")
                    if retry_count >= max_retries:
                        print("Max retries reached, exiting...")
                        break
                    time.sleep(1)  # Wait before retry
                    continue
                
                retry_count = 0  # Reset retry counter on successful frame read
                
                # Process frame and measure time
                start_time = time.time()
                processed_frame, num_faces = self.process_frame(frame)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Add FPS and stream info
                fps_text = f"FPS: {1/processing_time:.1f}"
                cv2.putText(processed_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                stream_type = "RTSP Stream" if parsed.scheme == 'rtsp' else "HTTP Stream"
                cv2.putText(processed_frame, stream_type, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame if output path is specified
                if output_path:
                    out.write(processed_frame)
                
                # Display frame
                cv2.imshow('Gender Detection - Stream', processed_frame)
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
    
        except KeyboardInterrupt:
            print("\nStream interrupted by user")
        except Exception as e:
            print(f"Error processing stream: {str(e)}")
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
        
        # Return processing statistics
        if processing_times:
            avg_fps = 1 / (sum(processing_times) / len(processing_times))
            return {
                'frames_processed': frame_count,
                'average_fps': avg_fps,
                'total_time': sum(processing_times)
            }
        return None
    
    def evaluate_model(self, val_loader):
        """Evaluate model on validation data"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def check_overfitting(self, train_loader, val_loader):
        """Compare performance on training and validation sets"""
        # Training performance
        train_loss, train_acc = self.evaluate_model(train_loader)
        
        # Validation performance
        val_loss, val_acc = self.evaluate_model(val_loader)
        
        # Calculate metrics
        loss_diff = abs(train_loss - val_loss)
        acc_diff = abs(train_acc - val_acc)
        
        results = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'loss_difference': loss_diff,
            'accuracy_difference': acc_diff,
            'is_overfitting': loss_diff > 0.1 and train_acc > val_acc
        }
        
        return results
    
    def print_overfitting_analysis(self, results):
        """Print overfitting analysis results"""
        print("\nModel Performance Analysis:")
        print(f"Training Loss: {results['train_loss']:.4f}")
        print(f"Training Accuracy: {results['train_accuracy']:.2f}%")
        print(f"Validation Loss: {results['val_loss']:.4f}")
        print(f"Validation Accuracy: {results['val_accuracy']:.2f}%")
        print(f"\nMetrics Difference:")
        print(f"Loss Difference: {results['loss_difference']:.4f}")
        print(f"Accuracy Difference: {results['accuracy_difference']:.2f}%")
        
        if results['is_overfitting']:
            print("\nWARNING: Model shows signs of overfitting!")
            print("- Training accuracy is significantly higher than validation accuracy")
            print("- Consider using regularization, dropout, or collecting more training data")
        else:
            print("\nModel shows no clear signs of overfitting.")

def main():
    # Create detector instance
    detector = GenderDetector()
    
    # Example usage menu
    while True:
        print("\nGender Detection Menu:")
        print("1. Real-time Webcam Detection")
        print("2. Image File Detection")
        print("3. Video File Detection")
        print("4. Stream Detection (RTSP/HTTP)")
        print("5. Check Model Overfitting")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
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
            stream_url = input("Enter stream URL (e.g., rtsp://username:password@ip:port/path or http://ip:port/path): ")
            save_output = input("Save output video? (y/n): ").lower() == 'y'
            
            output_path = None
            if save_output:
                output_path = input("Enter output video path (e.g., output.mp4): ")
            
            try:
                print("Starting stream detection... Press 'q' to quit")
                stats = detector.detect_from_stream(stream_url, output_path)
                if stats:
                    print(f"\nProcessing Statistics:")
                    print(f"Frames Processed: {stats['frames_processed']}")
                    print(f"Average FPS: {stats['average_fps']:.1f}")
                    print(f"Total Processing Time: {stats['total_time']:.1f} seconds")
                
            except Exception as e:
                print(f"Error processing stream: {str(e)}")
        
        elif choice == '5':
            print("Checking model for overfitting...")
            print("Note: You need to provide training and validation DataLoaders")
            print("This is a placeholder - implement data loading logic as needed")
            # Example usage (commented out as it needs DataLoader implementation):
            # results = detector.check_overfitting(train_loader, val_loader)
            # detector.print_overfitting_analysis(results)
            
        elif choice == '6':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()