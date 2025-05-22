import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import gc
import GPUtil

def print_gpu_utilization():
    GPUs = GPUtil.getGPUs()
    if GPUs:
        gpu = GPUs[0]
        print(f"GPU Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")

class GenderDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_dataset(dataset_path):
    image_paths = []
    labels = []
    
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.endswith((".jpg", ".png")):
                try:
                    gender = int(filename.split("_")[1])
                    if gender in [0, 1]:  # Validate gender label
                        img_path = os.path.join(root, filename)
                        image_paths.append(img_path)
                        labels.append(gender)
                except:
                    continue
    
    return image_paths, labels

def create_model(pretrained=True):
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(pretrained=pretrained)
    
    # Modify the classifier for binary classification
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_ftrs, 2)
    )
    return model

def plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_score, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(y_true, y_score, save_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
    plt.close()

def plot_metrics(y_true, y_pred, y_scores, save_dir='metrics'):
    os.makedirs(save_dir, exist_ok=True)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # Save Classification Report
    report = classification_report(y_true, y_pred, target_names=['Male', 'Female'])
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

def train_model(model, train_loader, val_loader, criterion, optimizer, timestamp, num_epochs=45, device='cuda'):
    model = model.to(device)
    best_acc = 0.0
    
    # Initialize mixed precision training
    scaler = GradScaler()
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/gender_detection_{timestamp}')
    
    # Create log file
    log_file = f'training_log_{timestamp}.csv'
    
    with open(log_file, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'GPU Memory Used'])
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Scale loss and call backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Get GPU memory usage
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            print_gpu_utilization()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('GPU Memory (MB)', gpu_memory, epoch)
        
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        all_labels = []
        all_preds = []
        all_scores = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/val', val_epoch_loss, epoch)
        writer.add_scalar('Accuracy/val', val_epoch_acc, epoch)
        
        print(f'Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        # Log the results to CSV
        with open(log_file, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch+1, epoch_loss, epoch_acc.item(), 
                           val_epoch_loss, val_epoch_acc.item(), gpu_memory])
        
        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_epoch_loss,
                'accuracy': val_epoch_acc,
            }, 'gender_detection_model.pt')
            
            # Generate and save metrics for best model
            plot_metrics(all_labels, all_preds, all_scores)
        
        # Garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    writer.close()
    return model

def export_to_onnx(model, sample_input, output_path):
    model.eval()
    torch.onnx.export(model,
                     sample_input,
                     output_path,
                     export_params=True,
                     opset_version=11,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})

if __name__ == "__main__":
    # Set device and optimize CUDA settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print_gpu_utilization()

    # Create timestamp for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    print("Loading dataset...")
    dataset_path = "UTKFace"
    image_paths, labels = load_dataset(dataset_path)
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # Create data loaders with pin_memory for faster GPU transfer
    train_dataset = GenderDataset(train_paths, train_labels, transform)
    val_dataset = GenderDataset(val_paths, val_labels, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64,  # Increased batch size for A10G
                            shuffle=True, num_workers=4,
                            pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64,
                          shuffle=False, num_workers=4,
                          pin_memory=True, persistent_workers=True)

    # Create and train model
    print("Creating model...")
    model = create_model(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, timestamp, num_epochs=45, device=device)

    # Export to ONNX
    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    export_to_onnx(model, dummy_input, 'gender_detection_model.onnx')
    
    print("Training complete! Models saved as:")
    print("- gender_detection_model.pt (PyTorch format)")
    print("- gender_detection_model.onnx (ONNX format)")
    print(f"Training logs saved in training_log_{timestamp}.csv")

    # Final GPU utilization
    if torch.cuda.is_available():
        print("\nFinal GPU Status:")
        print_gpu_utilization()