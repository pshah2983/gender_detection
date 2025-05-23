import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import torch
from model_evaluation import evaluate_model
import time

def plot_training_history(log_file):
    # Read training history
    df = pd.read_csv(log_file)
    
    # Create directory for visualizations
    os.makedirs('performance_analysis', exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300  # Higher resolution plots
    
    # Process all visualizations in parallel using subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Learning Curves (Loss and Accuracy)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df['Train Loss'], label='Training Loss', marker='o', markersize=4)
    ax1.plot(df['Val Loss'], label='Validation Loss', marker='o', markersize=4)
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 2. Accuracy Progress
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df['Train Acc'], label='Training Accuracy', marker='o', markersize=4)
    ax2.plot(df['Val Acc'], label='Validation Accuracy', marker='o', markersize=4)
    ax2.set_title('Accuracy Progress')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    # 3. Learning Rate Evolution
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(df['Learning Rate'], marker='o', markersize=4)
    ax3.set_title('Learning Rate Evolution')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    
    # 4. Training Progress with Generalization Gap
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(df['Train Acc'], label='Train Acc', color='blue', alpha=0.5)
    ax4.plot(df['Val Acc'], label='Val Acc', color='red', alpha=0.5)
    ax4.fill_between(range(len(df)), df['Train Acc'], df['Val Acc'], 
                    alpha=0.2, color='gray', label='Generalization Gap')
    
    # Add best validation accuracy marker
    best_epoch = df['Val Acc'].idxmax()
    best_acc = df['Val Acc'].max()
    ax4.scatter(best_epoch, best_acc, color='green', s=100, 
                label=f'Best Val Acc: {best_acc:.2f}%')
    
    ax4.set_title('Training Progress with Generalization Gap')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('performance_analysis/combined_training_metrics.png')
    plt.close()
    
    # Calculate and save performance statistics
    stats = {
        'Best Validation Accuracy': df['Val Acc'].max(),
        'Final Training Accuracy': df['Train Acc'].iloc[-1],
        'Final Validation Accuracy': df['Val Acc'].iloc[-1],
        'Lowest Training Loss': df['Train Loss'].min(),
        'Lowest Validation Loss': df['Val Loss'].min(),
        'Number of Epochs': len(df),
        'Final Learning Rate': df['Learning Rate'].iloc[-1],
        'Generalization Gap': df['Train Acc'].iloc[-1] - df['Val Acc'].iloc[-1],
        'Time to Best Accuracy': best_epoch + 1
    }
    
    with open('performance_analysis/training_stats.txt', 'w') as f:
        f.write("Training Performance Statistics\n")
        f.write("============================\n\n")
        for key, value in stats.items():
            if key == 'Number of Epochs' or key == 'Time to Best Accuracy':
                f.write(f"{key}: {int(value)}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")
        
        # Add convergence analysis
        f.write("\nConvergence Analysis:\n")
        f.write("===================\n")
        early_acc = df['Val Acc'].iloc[4]
        final_acc = df['Val Acc'].iloc[-1]
        improvement = final_acc - early_acc
        f.write(f"Early Stage Accuracy (Epoch 5): {early_acc:.2f}%\n")
        f.write(f"Final Accuracy: {final_acc:.2f}%\n")
        f.write(f"Total Improvement: {improvement:.2f}%\n")

def main():
    start_time = time.time()
    print("Starting comprehensive model evaluation and visualization...")
    
    # Create visualizations from training log
    log_file = 'logs/training_log_20250523_064611.csv'
    print("Generating training history visualizations...")
    plot_training_history(log_file)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set using L4 GPU...")
    test_data_path = 'new_dataset/Dataset/Test'
    results = evaluate_model(test_data_path)
    
    if results:
        print("\nTest Set Evaluation Results:")
        print(f"Accuracy: {results['accuracy']*100:.2f}%")
        print(f"ROC AUC Score: {results['roc_auc']:.4f}")
        print(f"\nDetailed metrics and visualizations have been saved to:")
        print("- metrics/ directory (test set evaluation results)")
        print("- performance_analysis/ directory (training history analysis)")
        print(f"\nTotal processing time: {(time.time() - start_time):.2f} seconds")
    else:
        print("Evaluation could not be completed. Please check the error messages above.")

if __name__ == "__main__":
    main()