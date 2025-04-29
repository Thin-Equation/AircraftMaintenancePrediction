"""
PyTorch implementations of various RNN-based models for aircraft maintenance prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNNModel(nn.Module):
    """Simple RNN model with single feature."""
    
    def __init__(self, sequence_length, features_dim, hidden_dim=1, output_dim=1, dropout_rate=0.2):
        """
        Initialize the RNN model.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        features_dim : int
            Number of features in the input data
        hidden_dim : int
            Size of the hidden dimension
        output_dim : int
            Number of output dimensions
        dropout_rate : float
            Dropout rate for regularization
        """
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=features_dim, 
                          hidden_size=hidden_dim, 
                          batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch_size, sequence_length, features_dim)
        out, _ = self.rnn(x)
        # Keep only the output of the last time step
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class MultiFeatureRNNModel(nn.Module):
    """Multi-feature RNN model with two RNN layers."""
    
    def __init__(self, sequence_length, features_dim, hidden_dim1=5, hidden_dim2=3, output_dim=1, dropout_rate=0.2):
        """
        Initialize the multi-feature RNN model.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        features_dim : int
            Number of features in the input data
        hidden_dim1 : int
            Size of the first hidden dimension
        hidden_dim2 : int
            Size of the second hidden dimension
        output_dim : int
            Number of output dimensions
        dropout_rate : float
            Dropout rate for regularization
        """
        super(MultiFeatureRNNModel, self).__init__()
        self.rnn1 = nn.RNN(input_size=features_dim, 
                           hidden_size=hidden_dim1, 
                           batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.rnn2 = nn.RNN(input_size=hidden_dim1, 
                           hidden_size=hidden_dim2, 
                           batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(hidden_dim2, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch_size, sequence_length, features_dim)
        out, _ = self.rnn1(x)
        out = self.dropout1(out)
        
        out, _ = self.rnn2(out)
        # Keep only the output of the last time step
        out = out[:, -1, :]
        out = self.dropout2(out)
        
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class BidirectionalRNNModel(nn.Module):
    """Bidirectional RNN model."""
    
    def __init__(self, sequence_length, features_dim, hidden_dim1=6, hidden_dim2=3, output_dim=1, dropout_rate=0.2):
        """
        Initialize the bidirectional RNN model.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        features_dim : int
            Number of features in the input data
        hidden_dim1 : int
            Size of the first hidden dimension
        hidden_dim2 : int
            Size of the second hidden dimension
        output_dim : int
            Number of output dimensions
        dropout_rate : float
            Dropout rate for regularization
        """
        super(BidirectionalRNNModel, self).__init__()
        self.brnn = nn.RNN(input_size=features_dim, 
                           hidden_size=hidden_dim1, 
                           batch_first=True,
                           bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # The output of bidirectional RNN has twice the hidden_dim size
        self.rnn = nn.RNN(input_size=hidden_dim1*2, 
                          hidden_size=hidden_dim2, 
                          batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(hidden_dim2, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch_size, sequence_length, features_dim)
        out, _ = self.brnn(x)
        out = self.dropout1(out)
        
        out, _ = self.rnn(out)
        # Keep only the output of the last time step
        out = out[:, -1, :]
        out = self.dropout2(out)
        
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class LSTMModel(nn.Module):
    """LSTM model for sequence prediction."""
    
    def __init__(self, sequence_length, features_dim, hidden_dim1=100, hidden_dim2=50, output_dim=1, dropout_rate=0.2):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        features_dim : int
            Number of features in the input data
        hidden_dim1 : int
            Size of the first hidden dimension
        hidden_dim2 : int
            Size of the second hidden dimension
        output_dim : int
            Number of output dimensions
        dropout_rate : float
            Dropout rate for regularization
        """
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=features_dim, 
                            hidden_size=hidden_dim1, 
                            batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(input_size=hidden_dim1, 
                            hidden_size=hidden_dim2, 
                            batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(hidden_dim2, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch_size, sequence_length, features_dim)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        out, _ = self.lstm2(out)
        # Keep only the output of the last time step
        out = out[:, -1, :]
        out = self.dropout2(out)
        
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class GRUModel(nn.Module):
    """GRU model for sequence prediction."""
    
    def __init__(self, sequence_length, features_dim, hidden_dim1=100, hidden_dim2=50, output_dim=1, dropout_rate=0.2):
        """
        Initialize the GRU model.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        features_dim : int
            Number of features in the input data
        hidden_dim1 : int
            Size of the first hidden dimension
        hidden_dim2 : int
            Size of the second hidden dimension
        output_dim : int
            Number of output dimensions
        dropout_rate : float
            Dropout rate for regularization
        """
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size=features_dim, 
                          hidden_size=hidden_dim1, 
                          batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.gru2 = nn.GRU(input_size=hidden_dim1, 
                          hidden_size=hidden_dim2, 
                          batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(hidden_dim2, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch_size, sequence_length, features_dim)
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        
        out, _ = self.gru2(out)
        # Keep only the output of the last time step
        out = out[:, -1, :]
        out = self.dropout2(out)
        
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


def create_model(model_name, sequence_length, features_dim, output_dim=1):
    """
    Factory function to create a PyTorch model based on model name.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to create (simple_rnn, multi_rnn, bidirectional, lstm, gru)
    sequence_length : int
        Length of input sequences
    features_dim : int
        Number of features in the input data
    output_dim : int, optional (default=1)
        Number of output dimensions
        
    Returns:
    --------
    torch.nn.Module
        The created PyTorch model
    """
    if model_name == 'simple_rnn':
        return SimpleRNNModel(sequence_length, features_dim, output_dim=output_dim)
    elif model_name == 'multi_rnn':
        return MultiFeatureRNNModel(sequence_length, features_dim, output_dim=output_dim)
    elif model_name == 'bidirectional':
        return BidirectionalRNNModel(sequence_length, features_dim, output_dim=output_dim)
    elif model_name == 'lstm':
        return LSTMModel(sequence_length, features_dim, output_dim=output_dim)
    elif model_name == 'gru':
        return GRUModel(sequence_length, features_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unsupported model: {model_name}")