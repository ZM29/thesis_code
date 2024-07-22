"""
This script implements the TSF with the Reversible Instance Normalization (RevIN) layer and the TSF architecture.
"""

from transformer_layer import ResidualBlockOutput, TSFTransformer
from input_layer import InputLayer
import torch.nn as nn
import torch

#setting device to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps = 1e-5, affine = True):
        """
        Class that implements the Reversible Instance Normalization. 
        This code is originally used in the 'Reversible Instance Normalization for Accurate Time-Series 
        Forecasting against Distribution Shift' study.

        Args:
            num_features (int): Number of features
            eps (float): A value added for numerical stability
            affine (bool): If True, RevIN has learnable affine parameter
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class TSF(nn.Module):
    def __init__(self, input_dim, model_dim, patch_size, max_len, num_layers, 
                 num_heads, hidden_dim, dropout_rate, output_horizon):
        super(TSF, self).__init__()
        """
        Class that implements the TSF architecture by combining all components.

        Args:
            input_dim (int): Dimension of input features
            model_dim (int): Dimension of the model
            patch_size (int): Size of each patch
            max_len (int): Maximum sequence length in PositionalEncoding
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            hidden_dim (int): Hidden dimension of the model
            dropout_rate (float): Dropout rate for regularization
            output_horizon (int): Time steps to forecast
        """
        self.input_layer = InputLayer(input_dim, model_dim, patch_size, max_len, hidden_dim, dropout_rate)
        self.transformer = TSFTransformer(num_layers, model_dim, num_heads)
        self.output_layer = ResidualBlockOutput(model_dim, hidden_dim, output_horizon * input_dim)
        self.output_horizon = output_horizon
        self.input_dim = input_dim

    def forward(self, numeric_data):
        
        #initialize RevIN and normalize input data
        rev_layer = RevIN(numeric_data.shape[0], affine=False)
        numeric_data = numeric_data.to(device)  
        numeric_data = rev_layer(numeric_data, 'norm')

        #processing the data through the TSF and denormalizing 
        numeric_embeddings, padding_mask = self.input_layer(numeric_data)
        hidden_states = self.transformer( numeric_embeddings, padding_mask)
        hidden_states = torch.cat(hidden_states, dim = 1)
        predictions = self.output_layer(hidden_states)
        predictions = rev_layer(predictions, 'denorm')

        #getting the last non-padded time step
        padding = torch.cat(padding_mask, dim = 1)
        last_index = padding.sum(dim = 1) - 1
        last_predictions = predictions[0][last_index[0]]
        return last_predictions
