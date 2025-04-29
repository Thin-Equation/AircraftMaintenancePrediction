#!/usr/bin/env python
"""
Training script for Aircraft Maintenance Prediction using PyTorch models.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch

# Add the parent directory to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data_processing.data_loader import load_data, get_feature_columns
from backend.data_processing.preprocessor import (
    add_remaining_useful_life, add_failure_within_window, normalize_data, prepare_test_data
)
from backend.data_processing.sequence import (
    create_sequence_dataset, create_labels
)
from backend.models.torch_models import create_model
from backend.utils.torch_trainer import ModelTrainer
from backend.visualization.plotting import plot_model_accuracy, plot_training_curve


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train aircraft maintenance prediction model with PyTorch.')
    
    parser.add_argument('--train', type=str, required=True, 
                        help='Path to training data file')
    parser.add_argument('--test', type=str, required=True, 
                        help='Path to test data file')
    parser.add_argument('--truth', type=str, required=True, 
                        help='Path to ground truth data file')
    parser.add_argument('--model', type=str, choices=['simple_rnn', 'multi_rnn', 'bidirectional', 'lstm', 'gru'],
                        default='lstm', help='Model architecture to train')
    parser.add_argument('--window', type=int, default=30, 
                        help='Window size in cycles for failure prediction')
    parser.add_argument('--sequence', type=int, default=50, 
                        help='Sequence length for RNN input')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=200, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--output', type=str, default='models', 
                        help='Directory to save model files')
    parser.add_argument('--single-feature', action='store_true', 
                        help='Use only sensor 2 as input feature')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA for training if available')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Set device
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    print(f"Loading data from {args.train}, {args.test}, and {args.truth}...")
    train_df, test_df, truth_df = load_data(args.train, args.test, args.truth)
    
    print("Preprocessing training data...")
    train_df = add_remaining_useful_life(train_df)
    train_df = add_failure_within_window(train_df, window_size=args.window)
    
    print("Preprocessing test data...")
    test_df = prepare_test_data(test_df, truth_df, window_size=args.window)
    
    print("Normalizing data...")
    train_df, test_df, scaler = normalize_data(train_df, test_df)
    
    # Select features for sequence creation
    if args.single_feature:
        feature_cols = ["s2"]
        print("Using single feature (s2) for training")
    else:
        feature_cols = get_feature_columns()
        print(f"Using {len(feature_cols)} features for training")
    
    print(f"Creating sequences with length {args.sequence}...")
    seq_set = create_sequence_dataset(train_df, args.sequence, feature_cols)
    label_set = create_labels(train_df, args.sequence, ['failure_within_w1'])
    
    print(f"Sequence dataset shape: {seq_set.shape}")
    print(f"Label set shape: {label_set.shape}")
    
    # Create validation split
    val_ratio = 0.05
    val_size = int(len(seq_set) * val_ratio)
    indices = np.random.permutation(len(seq_set))
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    X_train, y_train = seq_set[train_indices], label_set[train_indices]
    X_val, y_val = seq_set[val_indices], label_set[val_indices]
    
    # Create the model
    sequence_length = args.sequence
    features_dim = seq_set.shape[2]  # Number of features
    output_dim = label_set.shape[1]  # Number of outputs
    
    print(f"Creating {args.model} model...")
    model = create_model(args.model, sequence_length, features_dim, output_dim)
    
    # Set up trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        batch_size=args.batch,
        learning_rate=args.lr
    )
    
    # Train the model
    print(f"Training model for {args.epochs} epochs with batch size {args.batch}...")
    history = trainer.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        patience=10
    )
    
    # Evaluate on training data
    print("\nEvaluating on training data:")
    train_results = trainer.evaluate(X_train, y_train)
    print(f"Training Accuracy: {train_results['accuracy']:.4f}")
    print(f"Training Loss: {train_results['loss']:.4f}")
    print(f"Training Precision: {train_results['precision']:.4f}")
    print(f"Training Recall: {train_results['recall']:.4f}")
    print(f"Training F1 Score: {train_results['f1_score']:.4f}")
    
    # Save model
    model_path = os.path.join(args.output, f"{args.model}_pytorch.pth")
    trainer.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save training history
    history_df = pd.DataFrame({
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc']
    })
    history_path = os.path.join(args.output, f"{args.model}_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()