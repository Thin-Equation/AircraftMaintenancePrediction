#!/usr/bin/env python
"""
Evaluation script for Aircraft Maintenance Prediction using PyTorch models.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Add the parent directory to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data_processing.data_loader import load_data, get_feature_columns
from backend.data_processing.preprocessor import (
    add_remaining_useful_life, add_failure_within_window, normalize_data, prepare_test_data
)
from backend.data_processing.sequence import (
    get_last_sequence, get_last_labels
)
from backend.models.torch_models import create_model
from backend.utils.torch_trainer import ModelTrainer
from backend.utils.evaluation import (
    print_evaluation_metrics, plot_confusion_matrix, plot_prediction_comparison
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate aircraft maintenance prediction model with PyTorch.')
    
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the trained model file (.pth)')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['simple_rnn', 'multi_rnn', 'bidirectional', 'lstm', 'gru'],
                        help='Type of model architecture')
    parser.add_argument('--test', type=str, required=True, 
                        help='Path to test data file')
    parser.add_argument('--truth', type=str, required=True, 
                        help='Path to ground truth data file')
    parser.add_argument('--train', type=str, required=True, 
                        help='Path to training data file (needed for normalizing)')
    parser.add_argument('--window', type=int, default=30, 
                        help='Window size in cycles for failure prediction')
    parser.add_argument('--sequence', type=int, default=50, 
                        help='Sequence length for RNN input')
    parser.add_argument('--batch', type=int, default=200, 
                        help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='results', 
                        help='Directory to save evaluation results')
    parser.add_argument('--single-feature', action='store_true', 
                        help='Use only sensor 2 as input feature')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA for evaluation if available')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
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
        print("Using single feature (s2) for evaluation")
    else:
        feature_cols = get_feature_columns()
        print(f"Using {len(feature_cols)} features for evaluation")
    
    print(f"Preparing test sequences with length {args.sequence}...")
    last_sequences, valid_mask = get_last_sequence(test_df, args.sequence, feature_cols)
    last_labels = get_last_labels(test_df, 'failure_within_w1')
    
    # Create model
    sequence_length = args.sequence
    features_dim = last_sequences.shape[2]  # Number of features
    output_dim = last_labels.shape[1]  # Number of outputs
    
    print(f"Creating {args.model_type} model...")
    model = create_model(args.model_type, sequence_length, features_dim, output_dim)
    
    # Set up trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        batch_size=args.batch
    )
    
    # Load the model
    print(f"Loading model from {args.model}...")
    trainer.load_model(args.model)
    
    # Evaluate the model
    print("Evaluating model on test data...")
    results = trainer.evaluate(last_sequences, last_labels)
    
    # Print evaluation metrics
    print("\nTest Evaluation Results:")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Precision: {results['precision']:.4f}")
    print(f"Test Recall: {results['recall']:.4f}")
    print(f"Test F1 Score: {results['f1_score']:.4f}")
    print(f"Inference Time: {results['inference_time']:.2f} seconds")
    
    # Save confusion matrix plot
    cm_plot = plot_confusion_matrix(results['y_true'], results['y_pred'])
    cm_path = os.path.join(args.output, f"{args.model_type}_confusion_matrix.png")
    cm_plot.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save prediction comparison plot
    pred_plot = plot_prediction_comparison(results['y_true'], results['y_pred'])
    pred_path = os.path.join(args.output, f"{args.model_type}_prediction_comparison.png")
    pred_plot.savefig(pred_path)
    print(f"Prediction comparison saved to {pred_path}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'engine_id': test_df['id'].unique()[valid_mask],
        'actual': results['y_true'].flatten(),
        'predicted': results['y_pred'].flatten(),
        'correct': results['y_true'].flatten() == results['y_pred'].flatten()
    })
    
    csv_path = os.path.join(args.output, f"{args.model_type}_evaluation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to {csv_path}")
    
    # Save a summary of metrics
    metrics_df = pd.DataFrame({
        'accuracy': [results['accuracy']],
        'precision': [results['precision']],
        'recall': [results['recall']],
        'f1_score': [results['f1_score']],
        'inference_time': [results['inference_time']]
    })
    
    metrics_path = os.path.join(args.output, f"{args.model_type}_metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics summary saved to {metrics_path}")


if __name__ == "__main__":
    main()