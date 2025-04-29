"""
Functions for model evaluation in the Aircraft Maintenance Prediction project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score


def evaluate_model(model, x_test, y_test, batch_size=50, verbose=1):
    """
    Evaluate a model on test data.
    
    Parameters:
    -----------
    model : tensorflow.keras.models.Model
        The trained model to evaluate
    x_test : numpy.ndarray
        Test input data
    y_test : numpy.ndarray
        Test target data
    batch_size : int, optional (default=50)
        Batch size for evaluation
    verbose : int, optional (default=1)
        Verbosity mode
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Time the evaluation
    start = time.time()
    scores = model.evaluate(x_test, y_test, verbose=verbose, batch_size=batch_size)
    end = time.time()
    
    # Make predictions
    y_pred = (model.predict(x_test, verbose=verbose, batch_size=batch_size) > 0.5).astype("int32")
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Create results dictionary
    results = {
        'accuracy': scores[1],
        'loss': scores[0],
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'inference_time': end - start,
        'y_pred': y_pred,
        'y_true': y_test
    }
    
    return results


def plot_confusion_matrix(y_true, y_pred, figsize=(8, 6)):
    """
    Plot confusion matrix for binary classification.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    figsize : tuple, optional (default=(8, 6))
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object with the confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    return ax.get_figure()


def plot_prediction_comparison(y_true, y_pred, figsize=(10, 5)):
    """
    Plot comparison between true and predicted values.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    figsize : tuple, optional (default=(10, 5))
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object with the comparison plot
    """
    fig = plt.figure(figsize=figsize)
    plt.plot(y_pred, color="blue")
    plt.plot(y_true, color="green")
    plt.title('Prediction Comparison')
    plt.ylabel('Value')
    plt.xlabel('Sample')
    plt.legend(['Predicted', 'Actual Data'], loc='upper left')
    
    return fig


def print_evaluation_metrics(results):
    """
    Print the evaluation metrics in a readable format.
    
    Parameters:
    -----------
    results : dict
        Dictionary with evaluation results as returned by evaluate_model
    """
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Inference Time: {results['inference_time']:.2f} seconds")