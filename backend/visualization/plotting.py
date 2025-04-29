"""
Visualization utilities for plotting model performance and data insights.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_accuracy(history, width=10, height=5):
    """
    Plot model accuracy over training epochs.
    
    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        History object returned by model.fit()
    width : int, optional (default=10)
        Figure width
    height : int, optional (default=5)
        Figure height
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object with the accuracy plot
    """
    fig = plt.figure(figsize=(width, height))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    return fig


def plot_training_curve(history, width=10, height=5):
    """
    Plot model loss over training epochs.
    
    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        History object returned by model.fit()
    width : int, optional (default=10)
        Figure width
    height : int, optional (default=5)
        Figure height
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object with the loss plot
    """
    fig = plt.figure(figsize=(width, height))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    return fig


def plot_sensor_data(df, sensor_cols, engine_id=1, figsize=(20, 8)):
    """
    Plot sensor data for a specific engine.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the sensor data
    sensor_cols : list
        List of sensor column names
    engine_id : int, optional (default=1)
        ID of the engine to plot
    figsize : tuple, optional (default=(20, 8))
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object with the sensor data plot
    """
    engine_data = df[df.id == engine_id]
    fig = engine_data[sensor_cols].plot(figsize=figsize).get_figure()
    plt.title(f'Sensor Data for Engine ID {engine_id}')
    plt.xlabel('Cycles')
    plt.ylabel('Sensor Values')
    
    return fig


def plot_sensor_trend(df, sensor_col, engine_id=1, figsize=(10, 3)):
    """
    Plot a specific sensor trend for a specific engine.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the sensor data
    sensor_col : str
        Sensor column name
    engine_id : int, optional (default=1)
        ID of the engine to plot
    figsize : tuple, optional (default=(10, 3))
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object with the sensor trend plot
    """
    fig = df[df.id == engine_id][sensor_col].plot(figsize=figsize).get_figure()
    plt.title(f'{sensor_col} Trend for Engine ID {engine_id}')
    plt.xlabel('Cycles')
    plt.ylabel(f'{sensor_col} Value')
    
    return fig


def plot_rul_distribution(df, figsize=(10, 6)):
    """
    Plot the distribution of Remaining Useful Life (RUL) values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the RUL column
    figsize : tuple, optional (default=(10, 6))
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object with the RUL distribution plot
    """
    fig = plt.figure(figsize=figsize)
    sns.histplot(df['RUL'], bins=30, kde=True)
    plt.title('Distribution of Remaining Useful Life (RUL)')
    plt.xlabel('RUL (cycles)')
    plt.ylabel('Frequency')
    
    return fig


def plot_failure_distribution(df, figsize=(10, 6)):
    """
    Plot the distribution of failure labels.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the failure_within_w1 column
    figsize : tuple, optional (default=(10, 6))
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object with the failure distribution plot
    """
    fig = plt.figure(figsize=figsize)
    sns.countplot(x='failure_within_w1', data=df)
    plt.title('Distribution of Failure Labels')
    plt.xlabel('Failure within window (1=Yes, 0=No)')
    plt.ylabel('Count')
    
    return fig