"""
Functions for loading aircraft engine data from text files.
"""

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing


def load_data(train_path, test_path, truth_path):
    """
    Load and prepare training, test, and ground truth data.
    
    Parameters:
    -----------
    train_path : str
        Path to the training data file
    test_path : str
        Path to the test data file
    truth_path : str
        Path to the ground truth data file
    
    Returns:
    --------
    tuple
        (train_df, test_df, truth_df) as pandas DataFrames
    """
    # Load raw data
    train_df = pd.read_csv(train_path, sep=" ", header=None)
    test_df = pd.read_csv(test_path, sep=" ", header=None)
    truth_df = pd.read_csv(truth_path, sep=" ", header=None)
    
    # Drop columns with NaN values
    train_df.dropna(axis=1, inplace=True)
    test_df.dropna(axis=1, inplace=True)
    truth_df.dropna(axis=1, inplace=True)
    
    # Add column names
    cols_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                  's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                  's15', 's16', 's17', 's18', 's19', 's20', 's21']
    
    train_df.columns = cols_names
    test_df.columns = cols_names
    truth_df.columns = ['additional_rul']
    truth_df['id'] = truth_df.index + 1
    
    # Sort by id and cycle for proper sequencing
    train_df.sort_values(['id', 'cycle'], inplace=True)
    test_df.sort_values(['id', 'cycle'], inplace=True)
    
    return train_df, test_df, truth_df


def get_sensor_columns():
    """
    Get the names of sensor columns.
    
    Returns:
    --------
    list
        Names of sensor columns
    """
    return ['s' + str(i) for i in range(1, 22)]


def get_feature_columns():
    """
    Get complete list of feature columns used for modeling.
    
    Returns:
    --------
    list
        Names of all feature columns including settings and sensors
    """
    sensor_cols = get_sensor_columns()
    feature_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    feature_cols.extend(sensor_cols)
    return feature_cols