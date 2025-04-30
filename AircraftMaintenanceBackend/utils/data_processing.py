import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from typing import List, Tuple, Dict
import os
import sys

# Add the project root to the path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class DataProcessor:
    def __init__(self, train_path: str, test_path: str, truth_path: str, window_size: int = 30, sequence_length: int = 50):
        """
        Initialize the DataProcessor with file paths and parameters
        
        Args:
            train_path: Path to training data
            test_path: Path to testing data
            truth_path: Path to ground truth data
            window_size: Size of the window for failure prediction (default: 30)
            sequence_length: Length of sequences for input to models (default: 50)
        """
        self.train_path = train_path
        self.test_path = test_path
        self.truth_path = truth_path
        self.window_size = window_size
        self.sequence_length = sequence_length
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.train_df = None
        self.test_df = None
        self.truth_df = None
        self.cols_normalize = None
        
    def load_data(self) -> None:
        """Load and preprocess the datasets"""
        # Load datasets
        self.train_df = pd.read_csv(self.train_path, sep=" ", header=None)
        self.test_df = pd.read_csv(self.test_path, sep=" ", header=None)
        self.truth_df = pd.read_csv(self.truth_path, sep=" ", header=None)
        
        # Drop NaN columns
        self.train_df.dropna(axis=1, inplace=True)
        self.test_df.dropna(axis=1, inplace=True)
        self.truth_df.dropna(axis=1, inplace=True)
        
        # Add column names
        cols_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                      's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                      's15', 's16', 's17', 's18', 's19', 's20', 's21']
        
        self.train_df.columns = cols_names
        self.test_df.columns = cols_names
        
        # Sort values
        self.train_df.sort_values(['id', 'cycle'], inplace=True)
        self.test_df.sort_values(['id', 'cycle'], inplace=True)
        
    def process_training_data(self) -> None:
        """Process training data: calculate RUL and create target variable"""
        # Calculate RUL for training data
        rul = pd.DataFrame(self.train_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        self.train_df = self.train_df.merge(rul, on=['id'], how='left')
        self.train_df['RUL'] = self.train_df['max'] - self.train_df['cycle']
        self.train_df.drop('max', axis=1, inplace=True)
        
        # Generate label column for training data
        self.train_df['failure_within_w1'] = np.where(self.train_df['RUL'] <= self.window_size, 1, 0)
        
        # Normalize data
        self.train_df['cycle_norm'] = self.train_df['cycle']
        self.cols_normalize = self.train_df.columns.difference(['id', 'cycle', 'RUL', 'failure_within_w1'])
        
        norm_train_df = pd.DataFrame(
            self.min_max_scaler.fit_transform(self.train_df[self.cols_normalize]),
            columns=self.cols_normalize,
            index=self.train_df.index
        )
        
        join_df = self.train_df[['id', 'cycle', 'RUL', 'failure_within_w1']].join(norm_train_df)
        self.train_df = join_df.reindex(columns=self.train_df.columns)
        
    def process_test_data(self) -> None:
        """Process test data: calculate RUL and normalize"""
        # Normalize test data
        self.test_df['cycle_norm'] = self.test_df['cycle']
        
        norm_test_df = pd.DataFrame(
            self.min_max_scaler.transform(self.test_df[self.cols_normalize]),
            columns=self.cols_normalize,
            index=self.test_df.index
        )
        
        test_join_df = self.test_df[self.test_df.columns.difference(self.cols_normalize)].join(norm_test_df)
        self.test_df = test_join_df.reindex(columns=self.test_df.columns)
        self.test_df = self.test_df.reset_index(drop=True)
        
        # Calculate RUL for test data
        rul = pd.DataFrame(self.test_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        self.truth_df.columns = ['additional_rul']
        self.truth_df['id'] = self.truth_df.index + 1
        self.truth_df['max'] = rul['max'] + self.truth_df['additional_rul']
        self.truth_df.drop('additional_rul', axis=1, inplace=True)
        
        # Calculate RUL and create target
        self.test_df = self.test_df.merge(self.truth_df, on=['id'], how='left')
        self.test_df['RUL'] = self.test_df['max'] - self.test_df['cycle']
        self.test_df.drop('max', axis=1, inplace=True)
        self.test_df['failure_within_w1'] = np.where(self.test_df['RUL'] <= self.window_size, 1, 0)
    
    def _sequence_generator(self, feature_df: pd.DataFrame, seq_length: int, seq_cols: List[str]):
        """Generate sequences for RNN input"""
        feature_array = feature_df[seq_cols].values
        num_elements = feature_array.shape[0]
        
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield feature_array[start:stop, :]
    
    def _label_generator(self, label_df: pd.DataFrame, seq_length: int, label: List[str]):
        """Generate labels for sequences"""
        label_array = label_df[label].values
        num_elements = label_array.shape[0]
        return label_array[seq_length:num_elements, :]
            
    def generate_train_sequences(self, features: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training sequences and labels
        
        Args:
            features: List of feature column names to use (default: all sensor and setting values)
            
        Returns:
            Tuple of (sequences, labels) as PyTorch tensors
        """
        if features is None:
            # Use all sensor columns and settings by default
            sensor_cols = ['s' + str(i) for i in range(1, 22)]
            features = ['setting1', 'setting2', 'setting3', 'cycle_norm']
            features.extend(sensor_cols)
        
        # Generate sequences for each engine id
        seq_gen = (list(self._sequence_generator(
            self.train_df[self.train_df['id'] == id], 
            self.sequence_length, 
            features
        )) for id in self.train_df['id'].unique())
        
        # Convert to numpy array
        seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
        
        # Generate labels
        label_gen = [self._label_generator(
            self.train_df[self.train_df['id'] == id], 
            self.sequence_length, 
            ['failure_within_w1']
        ) for id in self.train_df['id'].unique()]
        
        label_array = np.concatenate(label_gen).astype(np.float32)
        
        # Convert to PyTorch tensors
        sequences = torch.tensor(seq_array, dtype=torch.float32)
        labels = torch.tensor(label_array, dtype=torch.float32)
        
        return sequences, labels
    
    def generate_test_sequences(self, features: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate test sequences for the last sequence_length cycles of each engine
        
        Args:
            features: List of feature column names to use
            
        Returns:
            Tuple of (sequences, labels) as PyTorch tensors
        """
        if features is None:
            # Use all sensor columns and settings by default
            sensor_cols = ['s' + str(i) for i in range(1, 22)]
            features = ['setting1', 'setting2', 'setting3', 'cycle_norm']
            features.extend(sensor_cols)
        
        # Get the last sequence for each engine
        last_test_seq = [
            self.test_df[self.test_df['id'] == id][features].values[-self.sequence_length:] 
            for id in self.test_df['id'].unique() 
            if len(self.test_df[self.test_df['id'] == id]) >= self.sequence_length
        ]
        last_test_seq = np.asarray(last_test_seq).astype(np.float32)
        
        # Get the labels for the test sequences
        y_mask = [len(self.test_df[self.test_df['id'] == id]) >= self.sequence_length for id in self.test_df['id'].unique()]
        last_test_label = self.test_df.groupby('id')['failure_within_w1'].nth(-1)[y_mask].values
        last_test_label = last_test_label.reshape(last_test_label.shape[0], 1).astype(np.float32)
        
        # Convert to PyTorch tensors
        sequences = torch.tensor(last_test_seq, dtype=torch.float32)
        labels = torch.tensor(last_test_label, dtype=torch.float32)
        
        return sequences, labels
    
    def prepare_single_input(self, engine_data: List[Dict]) -> torch.Tensor:
        """
        Prepare a single input sequence for prediction
        
        Args:
            engine_data: List of dictionaries containing sensor readings
            
        Returns:
            PyTorch tensor ready for model input
        """
        # Convert input data to DataFrame
        df = pd.DataFrame(engine_data)
        
        # Ensure required columns exist
        required_cols = ['cycle'] + [f's{i}' for i in range(1, 22)] + ['setting1', 'setting2', 'setting3']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Add normalized cycle column
        df['cycle_norm'] = df['cycle']
        
        # Apply the same normalization as training data
        features_to_normalize = self.cols_normalize.intersection(df.columns)
        if len(features_to_normalize) > 0:
            df[features_to_normalize] = self.min_max_scaler.transform(df[features_to_normalize])
        
        # Extract the last sequence_length cycles
        if len(df) >= self.sequence_length:
            df = df[-self.sequence_length:]
        else:
            # Pad with copies of the first row if not enough data
            padding = pd.concat([df.iloc[[0]]] * (self.sequence_length - len(df)), ignore_index=True)
            df = pd.concat([padding, df], ignore_index=True)
        
        # Prepare feature columns
        sensor_cols = ['s' + str(i) for i in range(1, 22)]
        features = ['setting1', 'setting2', 'setting3', 'cycle_norm']
        features.extend(sensor_cols)
        
        # Extract features and convert to tensor
        input_array = df[features].values.astype(np.float32)
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        return input_tensor
    
    def run_full_processing(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the full data processing pipeline and return training and testing data
        
        Returns:
            Tuple of (train_sequences, train_labels, test_sequences, test_labels)
        """
        self.load_data()
        self.process_training_data()
        self.process_test_data()
        
        # Generate sequences with all features
        train_sequences, train_labels = self.generate_train_sequences()
        test_sequences, test_labels = self.generate_test_sequences()
        
        return train_sequences, train_labels, test_sequences, test_labels