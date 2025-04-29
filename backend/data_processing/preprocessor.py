"""
Functions for preprocessing aircraft engine data.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing


def add_remaining_useful_life(train_df):
    """
    Add Remaining Useful Life (RUL) column to the training dataframe.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data frame
        
    Returns:
    --------
    pandas.DataFrame
        Training data with RUL column
    """
    # Get the maximum cycle for each engine id
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    
    # Merge with original dataframe
    train_df = train_df.merge(rul, on=['id'], how='left')
    
    # Calculate RUL
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)
    
    return train_df


def add_failure_within_window(df, window_size=30):
    """
    Add binary classification target variable indicating if failure will occur within the specified window.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with RUL column
    window_size : int, optional (default=30)
        Size of the window (in cycles) to predict failure
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'failure_within_w1' column
    """
    df['failure_within_w1'] = np.where(df['RUL'] <= window_size, 1, 0)
    return df


def normalize_data(train_df, test_df=None):
    """
    Normalize the data using MinMax scaling.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data
    test_df : pandas.DataFrame, optional
        Test data to normalize using same scaler as training data
        
    Returns:
    --------
    tuple
        (normalized_train_df, normalized_test_df, scaler)
        If test_df is None, returns (normalized_train_df, scaler)
    """
    # Add cycle_norm column
    train_df['cycle_norm'] = train_df['cycle']
    
    # Columns to exclude from normalization
    exclude_cols = ['id', 'cycle', 'RUL', 'failure_within_w1']
    cols_normalize = train_df.columns.difference(exclude_cols)
    
    # Initialize and fit the MinMax scaler
    scaler = preprocessing.MinMaxScaler()
    norm_train_data = pd.DataFrame(
        scaler.fit_transform(train_df[cols_normalize]),
        columns=cols_normalize,
        index=train_df.index
    )
    
    # Join the normalized columns with the excluded ones
    join_df = train_df[exclude_cols].join(norm_train_data)
    normalized_train_df = join_df.reindex(columns=train_df.columns)
    
    if test_df is not None:
        # Apply the same normalization to test data
        test_df['cycle_norm'] = test_df['cycle']
        exclude_test_cols = ['id', 'cycle', 'RUL', 'failure_within_w1']
        test_cols_normalize = test_df.columns.difference(exclude_test_cols)
        
        norm_test_data = pd.DataFrame(
            scaler.transform(test_df[test_cols_normalize]),
            columns=test_cols_normalize,
            index=test_df.index
        )
        
        test_join_df = test_df[exclude_test_cols].join(norm_test_data)
        normalized_test_df = test_join_df.reindex(columns=test_df.columns)
        
        return normalized_train_df, normalized_test_df, scaler
    
    return normalized_train_df, scaler


def prepare_test_data(test_df, truth_df, window_size=30):
    """
    Calculate RUL and add failure window label for test data using ground truth.
    
    Parameters:
    -----------
    test_df : pandas.DataFrame
        Test data frame
    truth_df : pandas.DataFrame
        Ground truth data with additional RUL information
    window_size : int, optional (default=30)
        Size of the window to predict failure
        
    Returns:
    --------
    pandas.DataFrame
        Test data with RUL and failure window columns
    """
    # Get the maximum cycle for each engine id in the test data
    max_cycles = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    max_cycles.columns = ['id', 'max']
    
    # Add 'max' column to truth_df representing the total number of cycles
    merged_truth = truth_df.merge(max_cycles, on=['id'], how='left')
    merged_truth['max'] = merged_truth['max'] + merged_truth['additional_rul']
    merged_truth.drop('additional_rul', axis=1, inplace=True)
    
    # Merge with test_df
    enhanced_test_df = test_df.merge(merged_truth, on=['id'], how='left')
    
    # Calculate RUL
    enhanced_test_df['RUL'] = enhanced_test_df['max'] - enhanced_test_df['cycle']
    enhanced_test_df.drop('max', axis=1, inplace=True)
    
    # Add failure window label
    enhanced_test_df['failure_within_w1'] = np.where(enhanced_test_df['RUL'] <= window_size, 1, 0)
    
    return enhanced_test_df