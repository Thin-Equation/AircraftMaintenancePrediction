"""
PyTorch training utilities for Aircraft Maintenance Prediction.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelTrainer:
    """Trainer class for PyTorch models."""
    
    def __init__(self, model, device=None, batch_size=200, learning_rate=0.001, weight_decay=0.0):
        """
        Initialize the trainer.
        
        Parameters:
        -----------
        model : torch.nn.Module
            PyTorch model to train
        device : torch.device, optional
            Device to use for training (default: None, which uses CUDA if available)
        batch_size : int, optional
            Batch size for training (default: 200)
        learning_rate : float, optional
            Learning rate for optimizer (default: 0.001)
        weight_decay : float, optional
            Weight decay for optimizer (default: 0.0)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Set training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=200, patience=10):
        """
        Train the PyTorch model.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training input data
        y_train : numpy.ndarray
            Training target data
        X_val : numpy.ndarray, optional
            Validation input data
        y_val : numpy.ndarray, optional
            Validation target data
        epochs : int, optional
            Maximum number of training epochs (default: 200)
        patience : int, optional
            Number of epochs to wait for improvement before early stopping (default: 10)
            
        Returns:
        --------
        dict
            Dictionary with training history
        """
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            has_validation = True
        else:
            has_validation = False
            
        # History dictionary to store metrics
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        
        # Training loop
        print(f"Training on {self.device}")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []
            
            for inputs, targets in train_loader:
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Save predictions for accuracy calculation
                train_loss += loss.item() * inputs.size(0)
                train_predictions.extend((outputs > 0.5).cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
            
            # Calculate epoch metrics
            train_loss /= len(train_loader.dataset)
            train_acc = accuracy_score(train_targets, train_predictions)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if has_validation:
                self.model.eval()
                val_loss = 0.0
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        
                        val_loss += loss.item() * inputs.size(0)
                        val_predictions.extend((outputs > 0.5).cpu().numpy())
                        val_targets.extend(targets.cpu().numpy())
                
                val_loss /= len(val_loader.dataset)
                val_acc = accuracy_score(val_targets, val_predictions)
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} | '
                          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    # Load best model
                    self.model.load_state_dict(best_model_state)
                    break
            else:
                # Print progress without validation
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} | '
                          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # If we have validation data but didn't early stop, load the best model
        if has_validation and best_model_state is not None and epoch == epochs-1:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test input data
        y_test : numpy.ndarray
            Test target data
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        # Convert numpy arrays to PyTorch tensors
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Create dataset and dataloader
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # Evaluation mode
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Time the inference
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                predictions = (outputs > 0.5).cpu().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy())
        
        inference_time = time.time() - start_time
        test_loss /= len(test_loader.dataset)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        
        # Create results dictionary
        results = {
            'loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'inference_time': inference_time,
            'y_pred': np.array(all_predictions),
            'y_true': np.array(all_targets)
        }
        
        return results
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Binary predictions (0 or 1)
        numpy.ndarray
            Probability predictions (0.0 to 1.0)
        """
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = outputs.cpu().numpy()
            predictions = (outputs > 0.5).cpu().numpy()
            
        return predictions, probabilities
    
    def save_model(self, path):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        path : str
            Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])