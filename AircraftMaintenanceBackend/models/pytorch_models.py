import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class BaseModel(nn.Module):
    """Base class for all models with common methods"""
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def save(self, path: str) -> None:
        """Save model state to path"""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load model state from path"""
        self.load_state_dict(torch.load(path))


class SimpleRNN(BaseModel):
    """Simple RNN model for sequence classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 5, num_layers: int = 1, 
                 dropout: float = 0.2, bidirectional: bool = False, output_dim: int = 1):
        """
        Initialize SimpleRNN model
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Size of hidden layer (default: 5)
            num_layers: Number of RNN layers (default: 1)
            dropout: Dropout probability (default: 0.2)
            bidirectional: Whether to use bidirectional RNN (default: False)
            output_dim: Dimension of output (default: 1 for binary classification)
        """
        super(SimpleRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust output size if bidirectional
        factor = 2 if bidirectional else 1
        
        # Dropout and output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, output_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # x shape: (batch_size, sequence_length, input_dim)
        
        # RNN output: output, hidden
        # output shape: (batch_size, sequence_length, hidden_dim * num_directions)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        _, hidden = self.rnn(x)
        
        # For bidirectional RNN, concatenate the last hidden state from both directions
        if self.bidirectional:
            # Concatenate the last hidden state from forward and backward pass
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # Use the last hidden state
            hidden = hidden[-1, :, :]
            
        # Apply dropout and then linear layer
        out = self.dropout_layer(hidden)
        out = self.fc(out)
        return self.activation(out)


class LSTMModel(BaseModel):
    """LSTM model for sequence classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 100, num_layers: int = 2, 
                 dropout: float = 0.2, output_dim: int = 1):
        """
        Initialize LSTM model
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Size of hidden layer (default: 100)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout probability (default: 0.2)
            output_dim: Dimension of output (default: 1 for binary classification)
        """
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout and output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # x shape: (batch_size, sequence_length, input_dim)
        
        # LSTM output: output, (hidden, cell)
        # output shape: (batch_size, sequence_length, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        # cell shape: (num_layers, batch_size, hidden_dim)
        _, (hidden, _) = self.lstm(x)
        
        # Take the output from the last layer
        out = hidden[-1, :, :]
        
        # Apply dropout and then linear layer
        out = self.dropout_layer(out)
        out = self.fc(out)
        return self.activation(out)


class BiRNNModel(BaseModel):
    """Bidirectional RNN model for sequence classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 6, num_layers: int = 1, 
                 dropout: float = 0.2, output_dim: int = 1):
        """
        Initialize BiRNN model
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Size of hidden layer (default: 6)
            num_layers: Number of BiRNN layers (default: 1)
            dropout: Dropout probability (default: 0.2)
            output_dim: Dimension of output (default: 1 for binary classification)
        """
        super(BiRNNModel, self).__init__()
        
        # Create a SimpleRNN with bidirectional=True for BiDirectional functionality
        self.bi_rnn = SimpleRNN(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,  # This makes it bidirectional
            output_dim=output_dim
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (delegate to SimpleRNN with bidirectional=True)"""
        return self.bi_rnn(x)


class GRUModel(BaseModel):
    """GRU model for sequence classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 100, num_layers: int = 2, 
                 dropout: float = 0.2, output_dim: int = 1):
        """
        Initialize GRU model
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Size of hidden layer (default: 100)
            num_layers: Number of GRU layers (default: 2)
            dropout: Dropout probability (default: 0.2)
            output_dim: Dimension of output (default: 1 for binary classification)
        """
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout and output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # x shape: (batch_size, sequence_length, input_dim)
        
        # GRU output: output, hidden
        # output shape: (batch_size, sequence_length, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        _, hidden = self.gru(x)
        
        # Take the output from the last layer
        out = hidden[-1, :, :]
        
        # Apply dropout and then linear layer
        out = self.dropout_layer(out)
        out = self.fc(out)
        return self.activation(out)


# Create a factory function to get model by name
def get_model(model_name: str, input_dim: int, **kwargs) -> BaseModel:
    """
    Factory function to create and return a model by name
    
    Args:
        model_name: Name of the model to create ('simple_rnn', 'lstm', 'birnn', or 'gru')
        input_dim: Number of input features
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        Instantiated model
    
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name.lower() == 'simple_rnn':
        return SimpleRNN(input_dim=input_dim, **kwargs)
    elif model_name.lower() == 'lstm':
        return LSTMModel(input_dim=input_dim, **kwargs)
    elif model_name.lower() == 'birnn':
        return BiRNNModel(input_dim=input_dim, **kwargs)
    elif model_name.lower() == 'gru':
        return GRUModel(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available models: 'simple_rnn', 'lstm', 'birnn', 'gru'")