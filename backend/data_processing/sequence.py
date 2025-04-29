"""
Functions for generating sequences for time-series analysis.
"""

import numpy as np
import pandas as pd


def sequence_generator(feature_df, seq_length, seq_cols):
    """
    Generator function that creates sequences of data.
    
    Parameters:
    -----------
    feature_df : pandas.DataFrame
        DataFrame containing features
    seq_length : int
        Length of the sequence to generate
    seq_cols : list
        List of column names to include in the sequence
        
    Yields:
    -------
    numpy.ndarray
        Array of shape (seq_length, len(seq_cols)) containing a single sequence
    """
    feature_array = feature_df[seq_cols].values
    num_elements = feature_array.shape[0]
    
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield feature_array[start:stop, :]


def create_sequence_dataset(df, seq_length, feature_cols, id_col='id'):
    """
    Create a dataset of sequences from original dataframe for all engine IDs.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with engine data
    seq_length : int
        Length of sequences to generate
    feature_cols : list
        List of feature column names to include
    id_col : str, optional (default='id')
        Name of the engine ID column
        
    Returns:
    --------
    numpy.ndarray
        Array of shape (num_sequences, seq_length, num_features)
    """
    # Generate sequences for each engine id
    seq_gen = (list(sequence_generator(df[df[id_col]==id], seq_length, feature_cols))
               for id in df[id_col].unique())
    
    # Concatenate sequences
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    
    return seq_array


def create_labels(df, seq_length, label_col, id_col='id'):
    """
    Create labels corresponding to each sequence.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with engine data
    seq_length : int
        Length of sequences used
    label_col : str or list
        Column name(s) of the label
    id_col : str, optional (default='id')
        Name of the engine ID column
        
    Returns:
    --------
    numpy.ndarray
        Array of labels
    """
    def label_generator(label_df, seq_length, label):
        label_array = label_df[label].values
        num_elements = label_array.shape[0]
        return label_array[seq_length:num_elements, :]
    
    # Convert label to list if it's not already
    if not isinstance(label_col, list):
        label_col = [label_col]
    
    # Generate labels for each engine id
    label_gen = [label_generator(df[df[id_col]==id], seq_length, label_col)
                 for id in df[id_col].unique()]
    
    # Concatenate labels
    label_array = np.concatenate(label_gen).astype(np.float32)
    
    return label_array


def get_last_sequence(test_df, sequence_length, feature_cols, id_col='id'):
    """
    Get the last sequence for each engine in the test set for prediction.
    
    Parameters:
    -----------
    test_df : pandas.DataFrame
        Test data frame
    sequence_length : int
        Length of the sequence
    feature_cols : list
        List of feature column names
    id_col : str, optional (default='id')
        Name of the engine ID column
        
    Returns:
    --------
    tuple
        (last_sequences, mask) where mask indicates which engine IDs had enough data
    """
    # Filter engine IDs with enough data for a sequence
    valid_ids = [id for id in test_df[id_col].unique() 
                 if len(test_df[test_df[id_col]==id]) >= sequence_length]
    
    # Create a mask indicating which engine IDs are valid
    mask = [len(test_df[test_df[id_col]==id]) >= sequence_length 
            for id in test_df[id_col].unique()]
    
    # Get the last sequence for each valid engine ID
    last_sequences = [test_df[test_df[id_col]==id][feature_cols].values[-sequence_length:] 
                      for id in valid_ids]
    
    return np.asarray(last_sequences).astype(np.float32), mask


def get_last_labels(test_df, label_col, id_col='id'):
    """
    Get the last label for each engine in the test set.
    
    Parameters:
    -----------
    test_df : pandas.DataFrame
        Test data frame
    label_col : str
        Column name of the label
    id_col : str, optional (default='id')
        Name of the engine ID column
        
    Returns:
    --------
    numpy.ndarray
        Array of last labels for each engine ID
    """
    last_labels = test_df.groupby(id_col)[label_col].nth(-1).values
    return last_labels.reshape(last_labels.shape[0], 1).astype(np.float32)