import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import torch.cuda.amp as amp
from tqdm import tqdm
import torch.backends.cudnn as cudnn

def setup_directories():
    """Create necessary directories for storing results"""
    dirs = ['metrics', 'models', 'logs']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir='metrics'):
    """Plot and save training curves"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_dir='metrics'):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    return cm

def train_model():
    # Setup directories
    setup_directories()
    
    # Set device and enable mixed precision training for L4 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = amp.GradScaler()
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable cuDNN auto-tuner and benchmarking
        cudnn.benchmark = True
        cudnn.deterministic = False
        # Empty CUDA cache
        torch.cuda.empty_cache()

    # Data transforms with augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    }

    # Dataset paths
    data_dir = 'new_dataset/Dataset'
    train_dir = os.path.join(data_dir, 'Train')
    val_dir = os.path.join(data_dir, 'Validation')

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])

    # Calculate optimal batch size based on GPU memory
    # L4 has 24GB VRAM, we'll use a larger batch size
    batch_size = 256  # Increased for L4 GPU

    # Create data loaders with optimized settings for L4 GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=8,  # Increased for better CPU utilization
        pin_memory=True,
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    # Load model with memory optimization
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device, memory_format=torch.channels_last)  # Optimize memory layout
    
    # Enable channels last memory format for better performance
    model = model.to(memory_format=torch.channels_last)

    # Loss function and optimizer with gradient clipping
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True
    )

    # Logging setup
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join('logs', f'training_log_{current_time}.csv')
    writer = SummaryWriter(f'runs/gender_detection_{current_time}')

    # CSV header
    csv_header = ['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 
                 'Learning Rate', 'Best Val Acc']
    
    with open(log_filename, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(csv_header)

    # Training parameters
    num_epochs = 30
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in train_bar:
            # Move inputs to channels last format for better performance
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision training
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}')
            for inputs, labels in val_bar:
                inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Store predictions for confusion matrix
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Save to CSV
        with open(log_filename, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                epoch + 1, train_loss, train_acc, 
                val_loss, val_acc, current_lr, best_val_acc
            ])

        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join('models', 'gender_detection_model.pt'))
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')

        # Update learning rate
        scheduler.step(val_loss)

        # Generate plots every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            plot_training_curves(train_losses, val_losses, train_accs, val_accs)
            if epoch == num_epochs - 1:
                classes = ['Female', 'Male']
                plot_confusion_matrix(all_val_labels, all_val_preds, classes)
                
                # Generate and save classification report
                report = classification_report(all_val_labels, all_val_preds, 
                                            target_names=classes)
                with open(os.path.join('metrics', 'classification_report.txt'), 'w') as f:
                    f.write(report)

        # Memory cleanup at the end of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Training logs saved to: {log_filename}")
    print("Visualization metrics saved in 'metrics' directory")

if __name__ == '__main__':
    train_model()