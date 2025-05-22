import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from gender_detection import create_model

def load_dataset(dataset_path):
    images = []
    labels = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # UTKFace dataset filename format: [age]_[gender]_[race]_[date&time].jpg
                # gender: 0 for male, 1 for female
                try:
                    gender = int(filename.split("_")[1])
                    # Validate gender label
                    if gender not in [0, 1]:
                        continue
                        
                    img_path = os.path.join(root, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (64, 64))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(gender)
                        if len(images) % 1000 == 0:  # Progress indicator
                            print(f"Loaded {len(images)} images...")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
    
    if not images:
        raise ValueError("No valid images found in the dataset path")
    
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Gender distribution - Male (0): {labels.count(0)}, Female (1): {labels.count(1)}")
        
    return np.array(images), np.array(labels)

def train_gender_model(dataset_path, model_save_path):
    # Load and preprocess dataset
    print("Loading dataset...")
    X, y = load_dataset(dataset_path)
    X = X.astype('float32') / 255.0
    y = to_categorical(y, 2)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile model
    print("Creating model...")
    model = create_model()
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train,
              batch_size=32,
              epochs=20,
              validation_data=(X_test, y_test))
    
    # Save model
    print("Saving model...")
    model.save(model_save_path)
    
    # Evaluate model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {score[1]*100:.2f}%")

if __name__ == "__main__":
    dataset_path = "UTKFace"  # Replace with your dataset path
    model_save_path = "gender_detection_model.h5"
    train_gender_model(dataset_path, model_save_path)