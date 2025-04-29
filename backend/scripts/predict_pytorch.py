#!/usr/bin/env python
"""
Prediction script for Aircraft Maintenance Prediction using PyTorch models.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch

# Add the parent directory to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data_processing.data_loader import get_feature_columns
from backend.data_processing.sequence import get_last_sequence
from backend.models.torch_models import create_model
from backend.utils.torch_trainer import ModelTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with trained PyTorch aircraft maintenance model.')
    
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the trained model file (.pth)')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['simple_rnn', 'multi_rnn', 'bidirectional', 'lstm', 'gru'],
                        help='Type of model architecture')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to input data file')
    parser.add_argument('--train', type=str, required=True, 
                        help='Path to original training data file (needed for normalizing)')
    parser.add_argument('--sequence', type=int, default=50, 
                        help='Sequence length for RNN input')
    parser.add_argument('--batch', type=int, default=200, 
                        help='Batch size for prediction')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                        help='Path to save prediction results')
    parser.add_argument('--single-feature', action='store_true', 
                        help='Use only sensor 2 as input feature')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for binary classification')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA for prediction if available')
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()
    
    # Set device
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # We only need the training data for normalization
    print(f"Loading training data from {args.train} for normalization...")
    train_df = pd.read_csv(args.train, sep=" ", header=None)
    train_df.dropna(axis=1, inplace=True)
    
    # Add column names
    cols_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                  's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                  's15', 's16', 's17', 's18', 's19', 's20', 's21']
    train_df.columns = cols_names
    
    # Load input data for prediction
    print(f"Loading input data from {args.data}...")
    input_df = pd.read_csv(args.data, sep=" ", header=None)
    input_df.dropna(axis=1, inplace=True)
    
    if input_df.shape[1] != train_df.shape[1]:
        print(f"Warning: Input data has {input_df.shape[1]} columns but training data has {train_df.shape[1]} columns.")
        print("Trying to adapt input data to match training data...")
        
        # Adjust columns if necessary
        if input_df.shape[1] < train_df.shape[1]:
            for i in range(input_df.shape[1], train_df.shape[1]):
                input_df[i] = 0  # Fill with zeros as placeholder
        elif input_df.shape[1] > train_df.shape[1]:
            input_df = input_df.iloc[:, :train_df.shape[1]]
    
    input_df.columns = cols_names
    
    # Add cycle_norm column and normalize data
    train_df['cycle_norm'] = train_df['cycle']
    input_df['cycle_norm'] = input_df['cycle']
    
    # Fit scaler on training data
    exclude_cols = ['id', 'cycle']
    cols_normalize = train_df.columns.difference(exclude_cols)
    
    print("Normalizing input data...")
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(train_df[cols_normalize])
    
    # Apply normalization to input data
    norm_input_data = pd.DataFrame(
        scaler.transform(input_df[cols_normalize]),
        columns=cols_normalize,
        index=input_df.index
    )
    
    join_df = input_df[exclude_cols].join(norm_input_data)
    normalized_input_df = join_df.reindex(columns=input_df.columns)
    
    # Select features for sequence creation
    if args.single_feature:
        feature_cols = ["s2"]
        print("Using single feature (s2) for prediction")
    else:
        feature_cols = get_feature_columns()
        print(f"Using {len(feature_cols)} features for prediction")
    
    print(f"Preparing sequences with length {args.sequence} for prediction...")
    # Get last sequence for each engine
    sequences, valid_mask = get_last_sequence(normalized_input_df, args.sequence, feature_cols)
    
    if len(sequences) == 0:
        print("Error: No valid sequences found. Make sure each engine has at least"
              f" {args.sequence} data points.")
        sys.exit(1)
    
    # Create model
    sequence_length = args.sequence
    features_dim = sequences.shape[2]  # Number of features
    output_dim = 1  # Binary classification
    
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
    
    print(f"Making predictions for {len(sequences)} engines...")
    # Make predictions
    predictions, probabilities = trainer.predict(sequences)
    
    # Create results DataFrame
    valid_ids = normalized_input_df['id'].unique()[valid_mask]
    results_df = pd.DataFrame({
        'engine_id': valid_ids,
        'probability': probabilities.flatten(),
        'failure_predicted': predictions.flatten()
    })
    
    # Add interpretation column
    results_df['interpretation'] = results_df['failure_predicted'].apply(
        lambda x: "Maintenance Required" if x == 1 else "No Maintenance Required"
    )
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total engines analyzed: {len(results_df)}")
    print(f"Engines predicted to require maintenance: {results_df['failure_predicted'].sum()}")
    print(f"Engines predicted to NOT require maintenance: {len(results_df) - results_df['failure_predicted'].sum()}")


if __name__ == "__main__":
    main()