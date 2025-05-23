import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import os
import seaborn as sns
from tqdm import tqdm
import time
import torch.backends.cudnn as cudnn

def evaluate_model(test_data_path):
    print("Starting model evaluation on test dataset...")
    start_time = time.time()
    
    # Set up GPU optimizations
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()  # Clear GPU memory
    
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # Initialize model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    
    # Load model weights
    checkpoint = torch.load('model/gender_detection_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Enable torch inference mode for better performance
    torch.inference_mode(True)
    
    # Data transforms with GPU memory format optimization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    test_dataset = datasets.ImageFolder(
        test_data_path,
        transform=transform
    )
    
    # Optimize batch size for L4 GPU (24GB VRAM)
    batch_size = 256  # Increased for L4 GPU
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,  # Increased for better CPU->GPU transfer
        pin_memory=True,  # Enable pinned memory for faster CPU->GPU transfer
        prefetch_factor=2,  # Prefetch next 2 batches
        persistent_workers=True,  # Keep workers alive between batches
    )
    
    total_images = len(test_dataset)
    print(f"\nProcessing {total_images} images in batches of {batch_size}...")
    
    # Evaluation metrics
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    # Process batches with progress bar
    with torch.inference_mode(), torch.cuda.amp.autocast():  # Enable mixed precision
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device, non_blocking=True)  # Non-blocking transfer
            labels = labels.to(device, non_blocking=True)
            
            # Get model predictions
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store results (transfer to CPU in batches)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean()
    classes = test_dataset.classes
    
    # Generate reports and visualizations
    os.makedirs('metrics', exist_ok=True)
    
    # Classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=classes)
    
    print("\nClassification Report:")
    print(report)
    
    with open('metrics/classification_report.txt', 'w') as f:
        f.write("Test Set Evaluation Results\n")
        f.write("==========================\n\n")
        f.write(f"Total images evaluated: {len(test_dataset)}\n")
        f.write(f"Processing time: {(time.time() - start_time):.2f} seconds\n\n")
        f.write(report)
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('metrics/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('metrics/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed metrics
    with open('metrics/detailed_metrics.txt', 'w') as f:
        f.write("Detailed Model Metrics\n")
        f.write("====================\n\n")
        f.write(f"Total images evaluated: {len(test_dataset)}\n")
        f.write(f"Overall accuracy: {accuracy*100:.2f}%\n")
        f.write(f"ROC AUC Score: {roc_auc:.4f}\n")
        f.write(f"Processing time: {(time.time() - start_time):.2f} seconds\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    print(f"\nEvaluation completed in {(time.time() - start_time):.2f} seconds")
    print(f"Overall accuracy: {accuracy*100:.2f}%")
    print("Detailed results have been saved to the metrics directory")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'processing_time': time.time() - start_time
    }