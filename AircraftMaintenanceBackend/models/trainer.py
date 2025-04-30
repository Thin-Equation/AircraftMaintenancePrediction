import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import time
import os
import sys

# Add the project root to the path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from AircraftMaintenanceBackend.models.pytorch_models import BaseModel


class ModelTrainer:
    """Class to train and evaluate PyTorch models for aircraft maintenance prediction"""
    
    def __init__(self, model: BaseModel, device: str = None):
        """
        Initialize trainer with model and device
        
        Args:
            model: PyTorch model to train
            device: Device to run training on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = None
        self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def setup_optimizer(self, optimizer_type: str = 'adam', lr: float = 0.001, 
                        weight_decay: float = 0, **kwargs):
        """
        Configure optimizer
        
        Args:
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            lr: Learning rate
            weight_decay: L2 penalty (weight decay)
            **kwargs: Additional arguments for optimizer
        """
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay, 
                **kwargs
            )
        elif optimizer_type.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay, 
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
            
    def setup_scheduler(self, scheduler_type: str = 'reduce_lr_on_plateau', **kwargs):
        """
        Configure learning rate scheduler
        
        Args:
            scheduler_type: Type of scheduler
            **kwargs: Additional arguments for scheduler
        """
        if not self.optimizer:
            raise ValueError("Optimizer must be set up before scheduler")
            
        if scheduler_type.lower() == 'reduce_lr_on_plateau':
            patience = kwargs.get('patience', 10)
            factor = kwargs.get('factor', 0.1)
            min_lr = kwargs.get('min_lr', 1e-6)
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                patience=patience,
                factor=factor,
                min_lr=min_lr
            )
        elif scheduler_type.lower() == 'step_lr':
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 0.1)
            
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 200, early_stopping_patience: int = 10,
              model_save_path: str = None) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            early_stopping_patience: Number of epochs to wait before early stopping
            model_save_path: Path to save the best model
            
        Returns:
            Dictionary containing training history
        """
        if self.optimizer is None:
            self.setup_optimizer()
            
        # Initialize early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Record statistics
                train_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total
            
            # Update learning rate scheduler if it exists
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if model_save_path:
                    self.save_model(model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        end_time = time.time()
        print(f"Total training time: {end_time - start_time:.2f} seconds")
        
        return self.history
    
    def save_model(self, path: str) -> None:
        """Save model to path"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'model_class': self.model.__class__.__name__,
            'model_params': {
                'input_dim': getattr(self.model, 'input_dim', None),
                'hidden_dim': getattr(self.model, 'hidden_dim', None),
                'num_layers': getattr(self.model, 'num_layers', None),
                'dropout': getattr(self.model, 'dropout', None)
            }
        }, path)
        
    def load_model(self, path: str) -> None:
        """Load model from path"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        test_loss = 0.0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
        test_loss = test_loss / len(test_loader.dataset)
        
        # Convert to numpy arrays for sklearn metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Calculate metrics
        accuracy = (predictions == true_labels).mean()
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        results = {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        return results
    
    def plot_training_history(self, figsize: Tuple[int, int] = (10, 5)) -> None:
        """
        Plot training history
        
        Args:
            figsize: Figure size as (width, height)
        """
        # Plot accuracy
        plt.figure(figsize=figsize)
        plt.plot(self.history['train_acc'])
        plt.plot(self.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        
        # Plot loss
        plt.figure(figsize=figsize)
        plt.plot(self.history['train_loss'])
        plt.plot(self.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        
    def plot_confusion_matrix(self, cm: np.ndarray, figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix array
            figsize: Figure size as (width, height)
        """
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
        
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the model
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Predictions tensor
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            predictions = (outputs > 0.5).float()
            return predictions


def prepare_data_loaders(X_train: torch.Tensor, y_train: torch.Tensor, 
                         X_val: Optional[torch.Tensor] = None, y_val: Optional[torch.Tensor] = None,
                         X_test: Optional[torch.Tensor] = None, y_test: Optional[torch.Tensor] = None,
                         batch_size: int = 200, val_split: float = 0.05) -> Dict[str, DataLoader]:
    """
    Prepare DataLoaders for training, validation, and testing
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        X_test: Test features (optional)
        y_test: Test labels (optional)
        batch_size: Batch size for DataLoaders
        val_split: Validation split ratio (if X_val and y_val are not provided)
        
    Returns:
        Dictionary of DataLoaders
    """
    result = {}
    
    # If validation set is not provided, split training data
    if X_val is None or y_val is None:
        val_size = int(len(X_train) * val_split)
        train_size = len(X_train) - val_size
        
        # Split randomly
        indices = torch.randperm(len(X_train))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        # Create training DataLoader
        train_dataset = TensorDataset(X_train_split, y_train_split)
        result['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        # Create training DataLoader from full training set
        train_dataset = TensorDataset(X_train, y_train)
        result['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation DataLoader
    val_dataset = TensorDataset(X_val, y_val)
    result['val'] = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create test DataLoader if test data provided
    if X_test is not None and y_test is not None:
        test_dataset = TensorDataset(X_test, y_test)
        result['test'] = DataLoader(test_dataset, batch_size=batch_size)
    
    return result